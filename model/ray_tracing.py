#  NeuralWarp  All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#    * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#    * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

import torch
import torch.nn as nn
from utils import rend_util
from . import volSDF_utils

class RayTracing(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.object_bounding_sphere = kwargs["object_bounding_sphere"]
        self.n_steps = kwargs["n_steps"]
        self.N1_vol = kwargs["N_samples_interval"]
        self.inverse_cdf_proportion = kwargs["inverse_cdf_proportion"]

        self.n_iterations = kwargs["n_iterations"]

        self.step = torch.zeros(1).cuda()

        self.opacity_approximator = volSDF_utils.OpacityApproximator(kwargs["n_steps"], kwargs["epsilon"],
                                                                     kwargs["n_iterations"],
                                                                     kwargs["bisection_iterations"])

    def forward(self,
                implicit_network,
                cam_loc,
                ray_directions):

        batch_size, num_pixels, _ = ray_directions.shape

        # interesections between rays and the bounding sphere of the object
        sphere_intersections, mask_intersect = rend_util.get_sphere_intersection(cam_loc, ray_directions, r=self.object_bounding_sphere)
        acc_start_dis = torch.clip(sphere_intersections[..., 0], min=0)
        acc_end_dis = sphere_intersections[..., 1]
        sdf = lambda x: implicit_network(x)[:, 0]

        pts_intervals, sdf_values, beta_final = self.opacity_approximator(sdf, cam_loc, ray_directions, acc_start_dis,
                                                                          acc_end_dis, beta=implicit_network.beta())

        occ_values = implicit_network.occupancy_function(sdf_values, pts_intervals, beta=beta_final)
        ray_points = self.volume_sampling_inverse_cdf(pts_intervals, occ_values, ray_directions, cam_loc)

        return ray_points

    def volume_sampling_inverse_cdf(self, pts_intervals, occ_values, ray_directions, cam_loc):
        Np = occ_values.shape[0]
        reg_occ = occ_values
        cumprod = torch.cumprod((1 - reg_occ), dim=1).roll(1, dims=1)
        cumprod[:, 0] = 1

        o = 1 - cumprod  # not normalized: o[:, -1] is mostly 1 but <1 for points without intersection
        # regularization with cdf of uniform distribution
        o = self.inverse_cdf_proportion * o + (1 - self.inverse_cdf_proportion) * pts_intervals / pts_intervals[:, -1:]
        o = o / o[:, -1:]  # now normalization will not be a problem for points without intersection

        bounds = torch.linspace(0, 1, self.N1_vol + 1).cuda()
        tmp = torch.rand(Np, self.N1_vol).cuda()
        t = tmp * bounds[:-1] + (1 - tmp) * bounds[1:]

        inv = torch.searchsorted(o, t, right=True)
        ext_dists = torch.cat((pts_intervals[:, 0:1] * torch.ones((Np, 1), device="cuda"),
                               pts_intervals,
                               pts_intervals[:, -1:] * torch.ones((Np, 1), device="cuda")), dim=-1)

        ext_o = torch.cat((-torch.ones((Np, 1), device="cuda"),
                           o,
                           2 * torch.ones((Np, 1), device="cuda")), dim=-1)

        o_inf = ext_o.take_along_dim(inv, 1)
        o_sup = ext_o.take_along_dim(inv + 1, 1)
        d_inf = ext_dists.take_along_dim(inv, 1)
        d_sup = ext_dists.take_along_dim(inv + 1, 1)

        denom = o_sup - o_inf
        dist = d_inf
        # linear interpolation for points where o_sup != o_inf
        lin_interp = denom > 1e-6
        dist[lin_interp] += ((t - o_inf) * (d_sup - d_inf))[lin_interp] / denom[lin_interp]

        points = cam_loc[None] + dist[:, :, None] * ray_directions[0, :, None, :]

        return points

