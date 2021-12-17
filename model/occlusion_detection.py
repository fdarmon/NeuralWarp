#  NeuralWarp  All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#    * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#    * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

from torch import nn
import torch
from torch.nn import functional as F
from .volSDF_utils import *


# adapted algorithm from volSDF to deal with an additional dimension: number of source images
class OcclusionDetector(nn.Module):
    def __init__(self, implicit_net, min_distance, init_samples, n_iterations, epsilon=0.1, clipping_distance=0):
        super().__init__()
        self.implicit_net = implicit_net
        self.init_samples = init_samples
        self.nb_iter = n_iterations
        self.min_distance = min_distance
        self.epsilon = epsilon

    def upsample_T(self, sdf_vals, delta, intervals_dist, beta, error_ok):
        # add n new samples in the intervals with the most error (line 4 of algo)
        device = sdf_vals.device
        Np, Nsrc, Ns = sdf_vals.shape
        d_star = get_d_star(sdf_vals, delta)
        sigma = volSDF_sigma(sdf_vals, 1 / beta)
        int_error = approx_error_intervals(d_star, sigma, delta, beta)
        int_error = torch.clamp(int_error, max=100)
        error_prop = self.init_samples * int_error / (
                    int_error.sum(dim=2, keepdim=True) + 1e-6)  # (float) number of points to add for each interval

        # first add the floor of error_prop then randomly add points up to n
        floor_error_prop = torch.floor(error_prop)
        nb_points_to_add = floor_error_prop.long()
        candidates = torch.topk(error_prop - floor_error_prop, k=min(Ns - 1, self.init_samples), sorted=True, dim=-1).indices
        valid_candidates = torch.arange(min(Ns - 1, self.init_samples), device=device)[None].expand(Np, Nsrc, -1) < (
                    self.init_samples - torch.sum(nb_points_to_add, dim=-1, keepdim=True))

        first_idx = torch.arange(Np, device=device)[:, None, None].expand_as(valid_candidates)[valid_candidates]
        second_idx = torch.arange(Nsrc, device=device)[None, :, None].expand_as(valid_candidates)[valid_candidates]
        third_idx = candidates[valid_candidates]
        nb_points_to_add[first_idx, second_idx, third_idx] += 1

        nb_points_to_add[:, :, 0] += (self.init_samples - nb_points_to_add.sum(
            dim=-1))  # manually add points in the first bin, never needed except when i == 0 and int_error = 0
        nb_points_to_add = nb_points_to_add.view(-1)

        # create new interval dist by adding the right number of points
        # tricky way to add efficiently new points inside the intervals with torch

        nb_points_plus_1 = nb_points_to_add + 1
        nb_tot = nb_points_plus_1.shape[0]
        origin_points = torch.arange(nb_tot, device=device).repeat_interleave(
            nb_points_plus_1, dim=0)

        nb_cumsum = torch.cumsum(nb_points_plus_1, dim=0)
        nb_repeated = nb_points_to_add.repeat_interleave(nb_points_plus_1)

        points_offset = nb_repeated + torch.arange(len(nb_repeated), device=device) - nb_cumsum.repeat_interleave(
            nb_points_plus_1) + 1
        points_offset = points_offset.reshape(Np, Nsrc, -1)
        origin_points = origin_points.reshape(Np, Nsrc, -1)
        nb_repeated = nb_repeated.reshape(Np, Nsrc, -1)

        new_intervals_dist = intervals_dist[:, :, :-1].reshape(-1).repeat_interleave(nb_points_plus_1).reshape(Np, Nsrc, -1)
        new_intervals_dist += points_offset * delta.view(-1)[origin_points] / (nb_repeated + 1)

        new_sdf_vals = sdf_vals[:, :, :-1].reshape(-1).repeat_interleave(nb_points_plus_1).reshape(Np, Nsrc, -1)
        new_sdf_vals += points_offset * (
                sdf_vals[..., 1:].reshape(-1)[origin_points] - sdf_vals[..., :-1].reshape(-1)[origin_points]) / (nb_repeated + 1)

        sdf_to_recompute = (points_offset > 0) & ~error_ok.unsqueeze(-1)

        # readd the last point of interval
        new_intervals_dist = torch.cat((new_intervals_dist, intervals_dist[:, :, -1:]), dim=-1)
        new_sdf_vals = torch.cat((new_sdf_vals, sdf_vals[:, :, -1:]), dim=-1)
        sdf_to_recompute = torch.cat((sdf_to_recompute, torch.zeros_like(intervals_dist[:, :, -1:], dtype=torch.bool)),
                                     dim=-1)

        return new_intervals_dist, new_sdf_vals, sdf_to_recompute

    def forward(self, cam_locs, intersection_points, in_src_im):

        sdf = lambda x: self.implicit_net(x)[:, 0]
        device = cam_locs.device
        Nc = cam_locs.shape[0]
        Np = intersection_points.shape[0]
        Ns = self.init_samples
        beta = self.implicit_net.beta()

        # remove batch dimension never used legacy
        rays = intersection_points.unsqueeze(1) - cam_locs.unsqueeze(0)
        max_dist = torch.norm(rays, dim=-1) - self.min_distance
        ray_directions = F.normalize(rays, dim=-1)
        sdf_to_compute = in_src_im.unsqueeze(-1).expand(Np, Nc, Ns)

        for i in range(self.nb_iter+1):
            if i == 0:
                min_dist = torch.zeros((1, 1), device=device)
                intervals_dist = torch.linspace(0, 1, steps=self.init_samples, device=device).view(1, -1)
                pts_intervals = min_dist + intervals_dist * (max_dist-min_dist).unsqueeze(-1)
                points = cam_locs.unsqueeze(1) + pts_intervals.unsqueeze(-1) * ray_directions.unsqueeze(2)
                sdf_values = torch.zeros_like(points[..., 0])

            else:
                pts_intervals, sdf_values, sdf_to_compute = self.upsample_T(sdf_values, delta, pts_intervals,
                                                                               beta, error_ok)
                points = cam_locs.unsqueeze(1) + pts_intervals.unsqueeze(-1) * ray_directions.unsqueeze(2)

            sdf_to_compute = in_src_im.unsqueeze(-1) & sdf_to_compute
            sdf_values[sdf_to_compute] = sdf(points[sdf_to_compute])
            delta = pts_intervals[..., 1:] - pts_intervals[..., :-1]
            d_star = get_d_star(sdf_values, delta)
            sigma = volSDF_sigma(sdf_values, 1 / beta)

            # in OcclusionDetector we are only interesed in  last point error
            error_ok = approx_error_intervals(d_star, sigma, delta, beta)[..., -1] < self.epsilon

        occ = self.implicit_net.occupancy_function(sdf_values, pts_intervals)

        res = 1 - torch.prod(1 - occ, dim=-1)
        res[~in_src_im] = 0
        return res

