import argparse
import os
from pyhocon import ConfigFactory
import torch
from PIL import Image
import math
import numpy as np
import pickle
import json
from tqdm import tqdm
import trimesh
from datasets import scene_dataset
from skimage import morphology as morph
from evaluation import mesh_filtering
from pathlib import Path


def clean_mesh(args):
    torch.set_default_dtype(torch.float32)

    if args.dataset == "dtu":
        size = [1600, 1200]
        datadir = "DTU"
    else:
        size = [3072, 2048]
        datadir = "strecha"
    scan_id = args.scene

    eval_dataset = scene_dataset.SceneDataset(datadir, size[::-1], scan_id)

    scale_mat = eval_dataset.get_scale_mat()
    num_images = len(eval_dataset)
    K = eval_dataset.intrinsics_all
    pose = eval_dataset.pose_all
    masks = eval_dataset.org_object_masks

    evaldir = args.evaldir
    meshname = args.meshname.split(".ply")[0]


    print("dilation...")
    dilated_masks = list()
    for m in tqdm(masks):
        if args.no_masks:
            dilated_masks.append(torch.ones_like(m, device="cuda"))
        else:
            dilated_masks.append(torch.from_numpy(morph.binary_dilation(m.numpy(), np.ones((51, 51)))))

    masks = torch.stack(dilated_masks).cuda()

    pose = pose.cuda()
    cams = [
        K[:, :3, :3].cuda(),
        pose[:, :3, :3].transpose(2, 1),
        - pose[:, :3, :3].transpose(2, 1) @ pose[:, :3, 3:],
        torch.tensor([size for i in range(num_images)]).cuda().float()
    ]
    # Load raw mesh
    mesh = trimesh.load(
        '{0}/{1}.ply'.format(evaldir, meshname, args.suffix), 'ply')

    # Transform to world coordinates
    mesh.apply_transform(np.linalg.inv(scale_mat))
    mesh_filtering.mesh_filter(args, mesh, masks, cams) # inplace filtering
    mesh.apply_transform(scale_mat)

    # Taking the biggest connected component
    if args.one_cc:
        components = mesh.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=float)
        mesh = components[areas.argmax()]

    mesh.export(f'{evaldir}/filtered{meshname}{args.suffix}.ply', 'ply')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["strecha", "dtu"])
    parser.add_argument('--scene', type=str, default=None, help='If set, taken to be the scan id.')
    parser.add_argument("--evaldir")
    parser.add_argument("--meshname")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--one_cc", action="store_true", default=True)
    parser.add_argument("--no_one_cc", action="store_false", dest="one_cc")
    parser.add_argument("--filter_visible_triangles", action="store_true")
    parser.add_argument('--min_nb_visible', type=int, default=2)
    parser.add_argument("--no_masks", action="store_true")

    args = parser.parse_args()

    clean_mesh(args)
