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
from utils import utils_3D
from .ray_tracing import RayTracing
from .occlusion_detection import OcclusionDetector
from .networks import ImplicitNetwork, RenderingNetwork

from torch.nn import functional as F

class NeuralWarp(nn.Module):
    def __init__(self, conf):
        super().__init__()
        rend_conf = conf.get_config("rendering")

        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.bound_sphere = conf.get_int("bound_sphere")
        self.plane_dist_thresh = conf.get_float("plane_dist_thresh")
        self.background_color = conf.get_string("background_color")

        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))

        self.network_colors = False
        self.warped_colors = False
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **rend_conf)
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'), object_bounding_sphere=self.bound_sphere)
        self.device = torch.device("cuda")

        self.h_patch_size = None
        self.offsets = None
        self.z_axis = torch.tensor([0, 0, 1]).to(self.device).float()
        if "occlusion_detector" in conf:
            self.occlusion_detector = OcclusionDetector(self.implicit_network, **conf.get_config("occlusion_detector"))
        else:
            self.occlusion_detector = None

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        # override load_state_dict since we need to have the same implicit_network in both NeuralWarp attributes and
        # OcclusionDetector attributes
        super().load_state_dict(state_dict, strict)
        if self.occlusion_detector is not None:
            self.occlusion_detector.implicit_net.load_state_dict(self.implicit_network.state_dict())

    def forward(self, input):
        _, nimgs, _, h, w = input["rgb"].shape
        inv_poses = input["inverse_pose"]
        nsrc = nimgs - 1
        final_output = dict()

        ref_intrinsics = input["intrinsics"][:, 0] # keep the 1x4x4 tensor (only used in get_cam_params)
        src_intr = input["intrinsics"][0, 1:, :3, :3]
        inv_ref_intr = input["inverse_intrinsics"][0, 0, :3, :3]

        ref_pose = input["pose"][0, 0]
        inv_src_pose = inv_poses[0, 1:]
        inv_ref_pose = inv_poses[0, 0]

        relative_proj = inv_src_pose @ ref_pose
        R_rel = relative_proj[:, :3, :3]
        t_rel = relative_proj[:, :3, 3:]
        R_ref = inv_ref_pose[:3, :3]
        t_ref = inv_ref_pose[:3, 3:]

        src_img = input["rgb"][:, 1:]

        uv = input["uv"]
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, ref_pose, ref_intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape
        self.implicit_network.eval()

        # sample points along the camera ray (volSDF method)
        with torch.no_grad():
            sampled_points = self.ray_tracer(implicit_network=self.implicit_network,
                                                                                 cam_loc=cam_loc,
                                                                                 ray_directions=ray_dirs)

            sampled_dists = torch.norm(sampled_points - cam_loc, dim=-1)

        self.implicit_network.train()
        ray_dirs = ray_dirs.reshape(-1, 3)
        view_dirs = -ray_dirs

        N, Ns, _ = sampled_points.shape

        flat_sampled_points = sampled_points.view(-1, 3)
        output, g = self.implicit_network.gradient(flat_sampled_points)
        sdf = output[:, 0].view(N, Ns)

        occupancy = self.implicit_network.occupancy_function(sdf, sampled_dists)

        surface_grad = g[:, 0]
        normals = F.normalize(surface_grad, dim=1)

        feature_vectors = output[:, 1:]
        expanded_viewdir = view_dirs.view(N, 1, 3).expand(-1, Ns, -1).reshape(-1, 3)

        cumprod = torch.cumprod((1 - occupancy), dim=1).roll(1, dims=1)
        all_cumulated = cumprod[:, 0].clone()
        cumprod[:, 0] = 1

        alpha_value = occupancy * cumprod

        intersection_points = torch.sum(alpha_value.unsqueeze(-1) * sampled_points, dim=1)
        intersection_points += all_cumulated[:, None] * sampled_points[:, -1] # background point is the last point (i.e. intersection with world sphere)
        network_object_mask = (all_cumulated < 0.5)  # no intersection if background contribution is more than half

        # for visu only
        avg_normals = torch.sum(
            alpha_value[:, :Ns].unsqueeze(-1) * normals.view(N, Ns, 3), dim=1
        )
        avg_depth = torch.sum(alpha_value * sampled_dists, dim=1)

        # __________________________________________________________________________________________________________
        # __________________________________________ Volumetric rendering __________________________________________
        # __________________________________________________________________________________________________________
        if self.network_colors:
            sampled_rgb_val = self.rendering_network(flat_sampled_points, normals, expanded_viewdir, feature_vectors).view(1, N, Ns, 3)
            network_rgb_vals = torch.sum(alpha_value.unsqueeze(-1) * sampled_rgb_val, dim=2).transpose(0, 1)

            if self.background_color == "white": # if black, no need
                network_rgb_vals += torch.ones((N, 1, 3), device=self.device) * all_cumulated.unsqueeze(1).unsqueeze(1)
        else:
            network_rgb_vals = None

        # __________________________________________________________________________________________________________
        # ___________________________________________ Patch warping ________________________________________________
        # __________________________________________________________________________________________________________
        if self.warped_colors and self.h_patch_size is not None:
            with torch.no_grad():
                rot_normals = R_ref @ normals.unsqueeze(-1)
                points_in_ref = R_ref @ flat_sampled_points.unsqueeze(
                    -1) + t_ref  # points in reference frame coordinate system
                d1 = torch.sum(rot_normals * points_in_ref, dim=1).unsqueeze(
                    1)  # distance from the plane to ref camera center

                d2 = torch.sum(rot_normals.unsqueeze(1) * (-R_rel.transpose(1, 2) @ t_rel).unsqueeze(0),
                               dim=2)  # distance from the plane to src camera center
                valid_hom = (torch.abs(d1) > self.plane_dist_thresh) & (
                        torch.abs(d1 - d2) > self.plane_dist_thresh) & ((d2 / d1) < 1)

                valid_hom = valid_hom.view(N, Ns, nsrc)
                valid_hom_mask = torch.sum(alpha_value.unsqueeze(1) * valid_hom.transpose(1, 2).float(), dim=2)
                valid_hom_mask += torch.ones_like(valid_hom_mask) * all_cumulated.unsqueeze(1)

                d1 = d1.squeeze()
                sign = torch.sign(d1)
                sign[sign == 0] = 1
                d = torch.clamp(torch.abs(d1), 1e-8) * sign

                H = src_intr.unsqueeze(1) @ (
                        R_rel.unsqueeze(1) + t_rel.unsqueeze(1) @ rot_normals.view(1, N * Ns, 1, 3) / d.view(1,
                                                                                                             N * Ns,
                                                                                                             1, 1)
                ) @ inv_ref_intr.view(1, 1, 3, 3)

                # replace invalid homs with fronto-parallel homographies
                H_invalid = src_intr.unsqueeze(1) @ (
                        R_rel.unsqueeze(1) + t_rel.unsqueeze(1) @ self.z_axis.view(1, 1, 1, 3).expand(-1, N*Ns, -1, -1) / sampled_dists.view(1, N*Ns, 1, 1)
                ) @ inv_ref_intr.view(1, 1, 3, 3)
                tmp_m = ~valid_hom.view(-1, nsrc).t()
                H[tmp_m] = H_invalid[tmp_m]

            pixels = uv.view(N, 1, 2) + self.offsets.float()
            Npx = pixels.shape[1]
            grid, warp_mask_full = self.patch_homography(H, pixels)

            warp_mask_full = warp_mask_full & (grid[..., 0] < (w - self.h_patch_size)) & (grid[..., 1] < (h - self.h_patch_size)) & (grid >= self.h_patch_size).all(dim=-1)
            warp_mask_full = warp_mask_full.view(nsrc, N, Ns, Npx)

            grid = torch.clamp(utils_3D.normalize(grid, h, w), -10, 10)

            sampled_rgb_val = F.grid_sample(src_img.squeeze(0), grid.view(nsrc, -1, 1, 2), align_corners=True).squeeze(-1).transpose(1, 2)
            sampled_rgb_val = sampled_rgb_val.view(nsrc, N, Ns, Npx, 3)
            sampled_rgb_val[~warp_mask_full, :] = 0.5 # set pixels out of image to 0.5 (grey color)

            warping_mask = warp_mask_full.float().mean(dim=-1)
            warping_mask = torch.sum(alpha_value.unsqueeze(1) * warping_mask.permute(1, 0, 2).float(), dim=2)
            # add background for mask
            warping_mask += torch.ones_like(warping_mask) * all_cumulated.unsqueeze(1)

            warped_rgb_vals = torch.sum(
                alpha_value.unsqueeze(-1).unsqueeze(-1) * sampled_rgb_val, dim=2
            ).transpose(0, 1)

            if self.background_color == "white": # if black, no need
                warped_rgb_vals += torch.ones((N, nsrc, 1, 3), device=self.device) * all_cumulated.view(N, 1, 1, 1)

        # __________________________________________________________________________________________________________
        # ___________________________________________ Pixel warping ________________________________________________
        # __________________________________________________________________________________________________________
        elif self.warped_colors and self.h_patch_size is None:
            grid_px, in_front = self.project(flat_sampled_points, inv_src_pose[:, :3], src_intr)
            grid = utils_3D.normalize(grid_px.squeeze(0), h, w, clamp=10)

            warping_mask_full = (in_front.squeeze(0) & (grid < 1).all(dim=-1) & (grid > -1).all(dim=-1))

            sampled_rgb_vals = F.grid_sample(src_img.squeeze(0), grid.unsqueeze(1), align_corners=True).squeeze(2).transpose(1, 2)
            sampled_rgb_vals[~warping_mask_full, :] = 0.5  # set pixels out of image to grey
            sampled_rgb_vals = sampled_rgb_vals.view(nsrc, N, -1, 3)
            warped_rgb_vals = torch.sum(alpha_value.unsqueeze(-1) * sampled_rgb_vals,
                                         dim=2).transpose(0, 1)

            if self.background_color == "white": # if black, no need
                warped_rgb_vals += torch.ones((N, nsrc, 3), device=self.device) * all_cumulated.unsqueeze(1).unsqueeze(1)

            warping_mask_full = warping_mask_full.view(nsrc, N, -1).permute(1, 2, 0).float()
            warping_mask = torch.sum(alpha_value.unsqueeze(-1) * warping_mask_full,
                                         dim=1)
            warping_mask += torch.ones_like(warping_mask) * all_cumulated.unsqueeze(1)
            valid_hom_mask = None

        else:  # no color warping at all (volSDF setup)
            warped_rgb_vals = None
            warping_mask = None
            valid_hom_mask = None

        # __________________________________________________________________________________________________________
        # ____________________________________________ Eikonal loss ________________________________________________
        # __________________________________________________________________________________________________________
        # Eikonal loss is computed on two points:
        # (i) a random point along the ray (ii) a random point from the sampled points
        random_along_rays = cam_loc + 2 * self.bound_sphere * torch.rand(num_pixels,
                                                                         device=self.device)[:, None] * ray_dirs
        _, grad_eikonal = self.implicit_network.gradient(random_along_rays)
        eikonal_normals = torch.cat((
            grad_eikonal,
            torch.gather(surface_grad.view(N, Ns, 3), 1,
                         torch.randint(Ns, size=(num_pixels, 1, 1), device=self.device).expand(-1, -1, 3))
        ), dim=1)

        # __________________________________________________________________________________________________________
        # _________________________________________ Occlusion detection ____________________________________________
        # __________________________________________________________________________________________________________
        if self.warped_colors and self.occlusion_detector:
            with torch.no_grad():
                cam_centers = input["pose"][0, 1:, :3, 3]

                occ_mask = self.occlusion_detector(cam_centers, intersection_points,
                                                   network_object_mask.unsqueeze(-1) & (warping_mask > 0.5))

        else:
            occ_mask = None

        final_output["network_rgb_values"] = network_rgb_vals
        final_output["warped_rgb_values"] = warped_rgb_vals
        final_output["valid_hom_mask"] = valid_hom_mask
        final_output["warping_mask"] = warping_mask
        final_output["occlusion_mask"] = occ_mask
        final_output["eikonal_normals"] = eikonal_normals
        # for visu only
        final_output["network_object_mask"] = network_object_mask
        final_output["sampled_points"] = flat_sampled_points
        final_output["avg_normals"] = avg_normals
        final_output["depth"] = avg_depth
        final_output["intersection_points"] = intersection_points

        return final_output

    def project(self, points, pose, intr):
        xyz = (intr.unsqueeze(1) @ pose.unsqueeze(1) @ utils_3D.add_hom(points).unsqueeze(-1))[..., :3, 0]
        in_front = xyz[..., 2] > 0
        grid = xyz[..., :2] / torch.clamp(xyz[..., 2:], 1e-8)
        return grid, in_front

    def patch_homography(self, H, uv):
        N, Npx = uv.shape[:2]
        Nsrc = H.shape[0]
        H = H.view(Nsrc, N, -1, 3, 3)
        hom_uv = utils_3D.add_hom(uv)

        # einsum is 30 times faster
        # tmp = (H.view(Nsrc, N, -1, 1, 3, 3) @ hom_uv.view(1, N, 1, -1, 3, 1)).squeeze(-1).view(Nsrc, -1, 3)
        tmp = torch.einsum("vprik,pok->vproi", H, hom_uv).reshape(Nsrc, -1, 3)

        grid = tmp[..., :2] / torch.clamp(tmp[..., 2:], 1e-8)
        mask = tmp[..., 2] > 0
        return grid, mask

    def update_patch_size(self, h_patch_size):
        self.h_patch_size = h_patch_size
        if h_patch_size is not None:
            self.offsets = rend_util.build_patch_offset(self.h_patch_size, self.device)
