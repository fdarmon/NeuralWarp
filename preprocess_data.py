import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import colmap
from utils import read_write_colmap


def compute_src_imgs(images, points3d, R, t, min_triangulation_angle, nsrc):
    im_ids = list(images.keys())
    im_id_to_idx = {im_ids[i]: i for i in range(len(images))}

    adj_mat = np.zeros((len(images), len(images)), dtype=int)
    adj_mat_tri = np.zeros((len(images), len(images)), dtype=int)

    R_rel = R[None, :] @ R[:, None].transpose(0, 1, 3, 2) # N * N * 3 * 3 relative rotation matrix from i to j
    t_rel = t[None, :] - R_rel @ t[:, None] # N * N * 3 * 1 relative translation from i to j

    rel_opt_center = -(np.transpose(R_rel, (0, 1, 3, 2)) @ t_rel).squeeze(3) # N * N * 3

    for p in tqdm(points3d, desc="Compute src images"):
        point = points3d[p]
        im_idx = np.array([im_id_to_idx[im_id] for im_id in point.image_ids])

        proj_p = R @ point.xyz[None, :, None] + t
        ray1 = proj_p[:, None, :, 0]
        ray2 = proj_p.squeeze(-1) - rel_opt_center
        cos = np.clip(
            np.sum(ray1 * ray2, axis=-1) / np.linalg.norm(ray1, axis=-1) / np.linalg.norm(ray2, axis=-1), -1, 1)
        tri_angles = np.arccos(cos) / np.pi * 180  # N * N
        valid_mat = np.zeros((len(images), len(images)), dtype=bool)
        valid_mat[im_idx[None, :], im_idx[:, None]] = True

        update_mat = (tri_angles > min_triangulation_angle) & valid_mat

        adj_mat[im_idx[None, :], im_idx[:, None]] += 1
        adj_mat_tri[update_mat] += 1

    sel_ims = dict()

    for i, im in enumerate(images):
        nb_common_points = adj_mat[i].copy()
        nb_common_points[adj_mat_tri[i] < (0.75 * adj_mat[i])] = 0

        l = np.argsort(nb_common_points)[-(nsrc):].tolist()
        sel = [images[list(images.keys())[i]].name for i in l]
        sel_ims[images[im].name] = sel

    return sel_ims

def get_calib_epfl(file):
    with open(file) as f:
        lines = f.readlines()

    v = np.array([float(s) for lin in lines for s in lin.strip().split()])
    K = v[:9].reshape((3, 3))
    R = v[12:21].reshape((3, 3))
    c = v[21:24]
    s = v[24:]

    R = R.T
    t = -R @ c
    return K, R, t[:, None], s

def compute_intersection_point(R, t):
    cam_centers = -R.transpose(0, 2, 1) @ t

    optical_axis = (R.transpose(0, 2, 1) @ np.array([0, 0, 1])[:, None]).squeeze(-1) # n * 3

    # solve intersection points as least square AX = b
    M = np.eye(3) - optical_axis[:, :, None] @ optical_axis[:, None, :]
    A = np.sum(M, axis=0)
    b = np.sum(M @ cam_centers, axis=0).squeeze(-1)
    intersection_point = np.linalg.solve(A, b)

    return intersection_point

def compute_scaling_matrix(R, t, sphere_size):
    intrsc_pnt = compute_intersection_point(R, t)
    cam_centers = -R.transpose(0, 2, 1) @ t
    s = sphere_size / (1.1 * np.linalg.norm(cam_centers.squeeze(-1) - intrsc_pnt, axis=1).max())
    scale_mat = np.tile(np.eye(4), (R.shape[0], 1, 1))
    scale_mat[:, :3] /= s
    scale_mat[..., :3, 3] = intrsc_pnt

    return scale_mat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dtu", "epfl"])
    parser.add_argument("--sphere_size", default=3., type=float)
    parser.add_argument("--scene")
    args = parser.parse_args()
    scene = args.scene
    dataset = args.dataset

    if dataset == "dtu":
        scene_path = Path("data/DTU") / f"scan{scene}"
    else:
        scene_path = Path("data/epfl") / (scene+"_dense")

    if  dataset == "dtu":
        sparse_path = scene_path / "sparse"
        colmap.run_sparse(scene_path, dataset, sparse_path)
        cam, imgs, pts = read_write_colmap.read_model(sparse_path)
        K, R, t, sizes = colmap.get_calib_from_sparse(cam, imgs)
        res = compute_src_imgs(imgs, pts, R, t, 5, 20)
        img_order = [int(imgs[idx].name.split(".")[0]) for idx in imgs]
        nb_imgs = K.shape[0]

    else:
        nb_imgs = len([f for f in (scene_path / "urd").iterdir() if f.name.endswith(".png")])
        imgs = [str(i).zfill(4) + ".png" for i in range(nb_imgs)]
        res = dict()
        K, R, t, sizes = list(), list(), list(), list()
        for i in range(nb_imgs):
            K_i, R_i, t_i, s_i = get_calib_epfl(scene_path / "urd" / (str(i).zfill(4) + ".png.camera"))
            K.append(K_i)
            R.append(R_i)
            t.append(t_i)
            res[str(i).zfill(4) + ".png"] = [str(j).zfill(4) + ".png" for j in range(nb_imgs) if j != i]
            sizes.append(s_i)

        K, R, t = np.stack(K), np.stack(R), np.stack(t)
        img_order = list(range(nb_imgs))

    with open(scene_path / "pairs.txt", "w") as f:
        for im in sorted(res.keys()):
            f.write(im)
            f.write(" ")
            f.write(" ".join(res[im]))
            f.write("\n")

    if dataset == "epfl": # compute scaling matrix
        new_scale_mat = compute_scaling_matrix(R, t, sphere_size=args.sphere_size)
        camera_dict = dict()
        for id_colmap, idx in enumerate(img_order):
            P = np.eye(4)
            P[:3, :3] = K[id_colmap] @ R[id_colmap]
            P[:3, 3:] = K[id_colmap] @ t[id_colmap, None]
            camera_dict["world_mat_%d" % idx] = P
            camera_dict['scale_mat_%d' % idx] = new_scale_mat[id_colmap]
            w, h = sizes[id_colmap]
            camera_dict["camera_mat_%d" % idx] = np.array([[2 / w, 0, -1, 0] , [0, 2 / h, -1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        np.savez(scene_path / "cameras.npz", **camera_dict)

