#  NeuralWarp  All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#    * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#    * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

import shutil
import subprocess
import numpy as np


from PIL import Image
from .utils_3D import quat_to_rot, rot_to_quat
import sqlite3
import os
from utils import rend_util

colmap_path = "./colmap"

def read_colmap_dtb_id(path):
    dtb = sqlite3.connect(path / "database.db")
    cursor = dtb.cursor()

    images = dict()
    cameras = dict()

    cursor.execute("SELECT name, image_id, camera_id FROM images;")
    for row in cursor:
        images[row[0]] = row[1]
        cameras[row[0]] = row[1]

    cursor.close()
    dtb.close()

    return images, cameras

def compute_Kmatrix_colmap(params):
    return np.array([
        [params[0], 0, params[2]],
        [0, params[1], params[3]],
        [0, 0, 1]
    ])

def get_calib_from_sparse(cameras, images):
    K = np.array([compute_Kmatrix_colmap(cameras[images[idx].camera_id].params) for idx in images],
                 dtype=np.float32)
    heights = np.array([cameras[images[idx].camera_id].height for idx in images], dtype=np.float32)
    widths = np.array([cameras[images[idx].camera_id].width for idx in images], dtype=np.float32)
    R = quat_to_rot(np.array([images[idx].qvec for idx in images])).astype(np.float32)
    t = np.array([images[idx].tvec for idx in images], dtype=np.float32)[..., None]

    return K, R, t, np.stack((widths, heights), axis=1)


def run_sparse(scene_path, dataset, output_path=None, im_folder_name="image"):
    if output_path is None:
        output_path = scene_path / "sparse"
    else:
        output_path = output_path

    if output_path.exists():
        ans = input("sparse path already exists, delete it ? y/n")
        if ans != "y":
            exit(0)

        shutil.rmtree(output_path)

    image_folder = scene_path / im_folder_name

    output_path.mkdir(parents=True)
    subprocess.call([
        str(colmap_path), "feature_extractor",
        "--database_path", str(output_path / "database.db"),
        "--image_path", str(image_folder)
    ])

    images, cameras = read_colmap_dtb_id(output_path)
    lines_cam = list()
    lines_im = list()


    if dataset == "dtu":
        camera_dict = np.load(scene_path / "cameras.npz")
    elif dataset == "strecha":
        camera_dict = np.load(scene_path / "cameras_newscale.npz")

    for filename in images:
        w, h = Image.open(scene_path / im_folder_name / filename).size
        idx = int(filename.split(".")[0].split("_")[-1])

        if not dataset == "blended" :
            world_mat = camera_dict['world_mat_%d' % idx].astype(np.float32)

            P = world_mat[:3, :4]
            K, pose = rend_util.load_K_Rt_from_P(None, P)
            R = pose[:3, :3].T
            t = -R @ pose[:3, 3]

        else:
            with open(scene_path / "cams" / (filename[:-4] + "_cam.txt")) as f:
                lines = f.readlines()

            lines = [l.strip() for l in lines]

            # extrinsics: line [1,5), 4x4 matrix
            extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
            # intrinsics: line [7-10), 3x3 matrix
            K = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

            R, t = extrinsics[:3, :3], extrinsics[:3, 3]

        cam_id = cameras[filename]
        im_id = images[filename]

        q = rot_to_quat(R[None]).squeeze(0)

        lines_cam.append(
            f"{cam_id} PINHOLE {w} {h} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n"
        )

        q_str = " ".join(map(str, q))
        t_str = " ".join(map(str, t))

        lines_im.append(f"{im_id} {q_str} {t_str} {cam_id} {filename}\n\n")

    with open(output_path / "cameras.txt", 'w') as f:
        f.writelines(lines_cam)

    with open(output_path / "images.txt", "w") as f:
        f.writelines(lines_im)

    with open(output_path / "points3D.txt", "w") as f:
        pass

    # exhaustive matching
    subprocess.call([
        str(colmap_path), "exhaustive_matcher",
        "--database_path", str(output_path / "database.db")
    ])

    # point triangulations
    subprocess.call([
        str(colmap_path), "point_triangulator",
        "--database_path", str(output_path / "database.db"),
        "--image_path", str(image_folder),
        "--input_path", str(output_path),
        "--output_path", str(output_path),
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.tri_ignore_two_view_tracks", "0",
        "--Mapper.fix_existing_images", "1"
    ])

    os.remove(output_path / "images.txt")
    os.remove(output_path / "cameras.txt")
    os.remove(output_path / "points3D.txt")
