import argparse
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
from tqdm import tqdm
import utils.general as utils
from skimage import morphology as morph
from evaluation import get_mesh
from utils import colmap
from pathlib import Path
from datasets import scene_dataset
from model.neuralWarp import NeuralWarp

def extract_mesh(args):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(args.conf)

    exps_folder_name = args.exps_folder_name

    evals_folder_name = args.evals_folder_name

    expname = args.conf.split("/")[-1].split(".")[0]
    scan_id = args.scan_id
    if scan_id is not None:
        expname = expname + '_{0}'.format(scan_id)

    if args.timestamp == 'latest':
        if os.path.exists(os.path.join(exps_folder_name, expname)):
            timestamps = os.listdir(os.path.join(exps_folder_name, expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            else:
                timestamp = sorted(timestamps)[-1]
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = args.timestamp

    utils.mkdir_ifnotexists(evals_folder_name)
    expdir = os.path.join(exps_folder_name, expname)
    evaldir = os.path.join(evals_folder_name, expname)
    utils.mkdir_ifnotexists(evaldir)

    model = NeuralWarp(conf=conf.get_config('model'))
    if torch.cuda.is_available():
        model.cuda()

    dataset_conf = conf.get_config('dataset')
    if args.scan_id is not None:
        dataset_conf['scan_id'] = args.scan_id

    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')

    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(args.checkpoint) + ".pth"))
    model.load_state_dict(saved_model_state["model_state_dict"], strict=False)

    if "iteration" in saved_model_state:
        iteration = saved_model_state['iteration']
    else:
        iteration = saved_model_state['epoch']

    if args.skip_inference and os.path.exists('{0}/surface_world_coordinates_{1}{2}.ply'.format(evaldir, iteration, args.suffix)):
        print("Mesh already extracted and skip inference set to True")
    else:
        eval_dataset = scene_dataset(**dataset_conf)

        scale_mat = eval_dataset.get_scale_mat()
        num_images = len(eval_dataset)
        K = eval_dataset.intrinsics_all
        pose = eval_dataset.pose_all

        masks = eval_dataset.org_object_masks

        print("dilation...")
        dilated_masks = list()

        for m in tqdm(masks):
            if args.no_masks:
                dilated_masks.append(torch.ones_like(m, device="cuda"))
            else:
                dilated_masks.append(torch.from_numpy(morph.binary_dilation(m.numpy(), np.ones((51, 51)))))
        masks = torch.stack(dilated_masks).cuda()

        model.eval()

        with torch.no_grad():
            size = conf.dataset.img_res[::-1]
            pose = pose.cuda()
            cams = [
                K[:, :3, :3].cuda(),
                pose[:, :3, :3].transpose(2, 1),
                - pose[:, :3, :3].transpose(2, 1) @ pose[:, :3, 3:],
                torch.tensor([size for i in range(num_images)]).cuda().float()
            ]

            meshraw = get_mesh.get_surface_high_res_mesh(
                occ=lambda x: model.implicit_network(x)[:, 0], refine_bb=not args.no_refine_bb,
                resolution=args.resolution, cams=cams, masks=masks, bbox_size=args.bbox_size
            )
            # Transform to world coordinates
            meshraw.apply_transform(scale_mat)
            meshraw.export('{0}/surface_raw_{1}{2}.ply'.format(evaldir, iteration, args.suffix), 'ply')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str)
    parser.add_argument("--dataset", choices=["dtu", 'strecha'])
    parser.add_argument('--method', choices = ["ours", "colmap"], default="ours")
    parser.add_argument("--skip_inference", action="store_true", help="if mesh exists, skip the network inference")
    parser.add_argument('--exps_folder_name', type=str, default="exps", help='The experiments folder name.')
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiment timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--scan_id', type=str, default=None, help='If set, taken to be the scan id.')
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--no_refine_bb', action="store_true", help='Skip bounding box refinement')
    parser.add_argument("--no_masks", action="store_true")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--bbox_size", default=1., type=float)

    args = parser.parse_args()

    args.evals_folder_name = "evals"
    extract_mesh(args)
