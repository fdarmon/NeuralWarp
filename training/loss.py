#  NeuralWarp  All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#    * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#    * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils import utils_3D
from utils import rend_util
from utils.ssim import SSIM

class Loss(nn.Module):
    def __init__(self, warping_weight=0, network_weight=0, eikonal_regularization=None,
                 h_patch_size=5, min_visibility=1e-3, patch_loss="ssim"):

        super().__init__()
        self.warping_weight = warping_weight
        self.network_weight = network_weight
        self.eikonal_regularization = eikonal_regularization
        self.ssim = SSIM(h_patch_size).cuda()
        self.patch_offset = rend_util.build_patch_offset(h_patch_size, torch.device("cuda"))

        self.min_visibility = min_visibility
        self.patch_loss = patch_loss

    def full_rgb_loss(self, rgb_values, rgb_gt):
        npx, nsrc, _ = rgb_values.shape
        rgb_gt = rgb_gt.reshape(npx, 1, 3)
        rgb_values = rgb_values.reshape(npx, nsrc, 3)
        rgb_loss = torch.sum(torch.abs(rgb_values - rgb_gt)) / nsrc
        return rgb_loss

    def masked_pixel_loss(self,rgb_values, rgb_gt, warp_mask):
        npx, nsrc, _ = rgb_values.shape

        if warp_mask.sum() == 0:
            return torch.tensor(0.0).cuda().float(), torch.ones_like(warp_mask[:, 0])

        warp_mask = warp_mask.float()

        num = torch.sum(warp_mask.unsqueeze(2) * torch.abs(rgb_values - rgb_gt.unsqueeze(1)), dim=1).sum(dim=1)
        denom = torch.sum(warp_mask, dim=1)

        valids = denom > self.min_visibility
        rgb_loss = torch.sum(num[valids] / denom[valids])

        return rgb_loss, valids

    def masked_patch_loss(self,rgb_values, rgb_gt, warp_mask):
        npx, nsrc, npatch, _ = rgb_values.shape

        warp_mask = warp_mask.float()

        if self.patch_loss == "l1":
            num = torch.sum(warp_mask.unsqueeze(-1).unsqueeze(-1) * torch.abs(rgb_values - rgb_gt.unsqueeze(1)),
                            dim=1).sum(dim=1).sum(dim=1) / npatch

        elif self.patch_loss == "ssim":
            num = torch.sum(warp_mask * self.ssim(rgb_values, rgb_gt), dim=1)

        else:
            raise NotImplementedError("Patch loss + " + self.patch_loss)

        denom = torch.sum(warp_mask, dim=1)

        valids = denom > self.min_visibility
        return torch.sum(num[valids] / denom[valids]), valids

    def forward(self, model_input, model_outputs):
        imgs = model_input['rgb']

        _, nimgs, _, h, w = imgs.shape

        res = {}
        loss = 0

        warped_rgb_val = model_outputs[f"warped_rgb_values"]
        patch_loss = (warped_rgb_val is not None) and (len(warped_rgb_val.shape) == 4)

        warp_mask = model_outputs[f"warping_mask"]
        uv = model_input[f"uv"]
        num_pixels = uv.shape[1]
        i, j = uv[0, :, 1].long(), uv[0, :, 0].long()
        if patch_loss:
            rgb_patches_gt = imgs[0, 0, :, i.unsqueeze(1) + self.patch_offset[..., 1],
                                  j.unsqueeze(1)+ self.patch_offset[..., 0]].permute(1, 2, 0)

        rgb_gt = imgs[:, 0, :, i, j].view(3, -1).t()

        if model_outputs[f"network_rgb_values"] is not None:
            network_rgb_loss = self.full_rgb_loss(model_outputs[f'network_rgb_values'], rgb_gt) / num_pixels
            res[f"network_rgb_loss"] = network_rgb_loss
            loss += self.network_weight * network_rgb_loss

        if warped_rgb_val is not None:
            occlusion_mask = model_outputs[f"occlusion_mask"]
            if occlusion_mask is not None:
                mask = warp_mask  * (1 - occlusion_mask)
            else:
                mask = warp_mask

            if patch_loss:
                mask = mask * model_outputs["valid_hom_mask"]
                warped_rgb_loss, valid_warp = self.masked_patch_loss(warped_rgb_val, rgb_patches_gt, mask)
            else:
                warped_rgb_loss, valid_warp = self.masked_pixel_loss(warped_rgb_val, rgb_gt, mask)

            warped_rgb_loss /= num_pixels

            res[f"warped_rgb_loss"] = warped_rgb_loss
            loss += self.warping_weight * warped_rgb_loss

        if self.eikonal_regularization:
            normals = model_outputs[f"eikonal_normals"]
            res[f"eikonal_loss"] = torch.sum((torch.norm(normals, dim=-1) - 1) ** 2) / normals.shape[1] / num_pixels
            loss += self.eikonal_regularization * res[f"eikonal_loss"]

        res["loss"] = loss

        return res