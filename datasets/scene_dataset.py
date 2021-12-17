#  NeuralWarp  All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#    * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#    * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util

class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""
    def __init__(self,
                 data_dir,
                 img_res,
                 scene=0,
                 nsrc=0,
                 h_patch_size=None,
                 uv_down=None,
                 ):

        if data_dir == "DTU":
            self.instance_dir = os.path.join('data', data_dir, 'scan{0}'.format(scene))
            im_folder_name = "image"
        else:
            # epfl
            self.instance_dir = os.path.join('data', data_dir, scene + "_dense")
            im_folder_name = "urd"

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        self.generator = torch.Generator()
        self.generator.manual_seed(np.random.randint(1e9))

        assert os.path.exists(self.instance_dir), "Data directory is empty" + str(self.instance_dir)

        self.sampling_idx = None
        self.small_uv = uv_down is not None
        self.uv_down = uv_down
        if self.small_uv:
            self.plot_img_res = img_res[0] // self.uv_down, img_res[1] // self.uv_down
        else:
            self.plot_img_res = img_res

        image_dir = '{0}/{1}'.format(self.instance_dir, im_folder_name)
        image_paths = sorted(utils.glob_imgs(image_dir))
        mask_dir = '{0}/mask'.format(self.instance_dir)
        mask_paths = sorted(utils.glob_imgs(mask_dir))

        self.n_images = len(image_paths)
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)

        camera_dict = np.load(self.cam_file)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for world_mat, scale_mat in zip(world_mats, scale_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = torch.stack(self.intrinsics_all)
        self.inv_intrinsics_all = torch.inverse(self.intrinsics_all)
        self.pose_all = torch.stack(self.pose_all)
        self.inv_pose_all = torch.inverse(self.pose_all)

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            self.rgb_images.append(torch.tensor(rgb).float())

        self.rgb_images = torch.stack(self.rgb_images)

        self.object_masks = []
        self.org_object_masks = []
        for path in mask_paths:
            object_mask = rend_util.load_mask(path)
            self.org_object_masks.append(torch.tensor(object_mask).bool())
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).bool())

        if nsrc == 'max':
            self.nsrc = self.rgb_images.shape[0] - 1
        else:
            self.nsrc = nsrc - 1
        with open(os.path.join(self.instance_dir, "pairs.txt")) as f:
            pairs = f.readlines()

        self.src_idx = []
        for p in pairs:
            splitted = p.split()[1:]  # drop the first one since it is the ref img
            fun = lambda s: int(s.split(".")[0])
            self.src_idx.append(torch.tensor(list(map(fun, splitted))))

        self.h_patch_size = h_patch_size


    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        if self.sampling_idx is None and self.small_uv:
            uv = uv[:, ::self.uv_down, ::self.uv_down]
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
        }

        src_idx = self.src_idx[idx][torch.randperm(len(self.src_idx[idx]),
                                                   generator=self.generator)][:self.nsrc]

        idx_list = torch.cat([torch.tensor(idx).unsqueeze(0), src_idx], dim=0)

        sample["pose"] = self.pose_all[idx_list]
        sample["inverse_pose"] = self.inv_pose_all[idx_list]
        sample["intrinsics"] = self.intrinsics_all[idx_list]
        sample["inverse_intrinsics"] = self.inv_intrinsics_all[idx_list]
        sample["idx_list"] = idx_list

        if self.sampling_idx is not None:
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            if self.h_patch_size:
                idx_img = torch.arange(self.total_pixels).view(self.img_res[0], self.img_res[1])
                if self.h_patch_size > 0:
                    idx_img = idx_img[self.h_patch_size:-self.h_patch_size, self.h_patch_size:-self.h_patch_size]
                idx_img = idx_img.reshape(-1)
                self.sampling_idx = idx_img[torch.randperm(idx_img.shape[0])[:sampling_size]]
            else:
                self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']
