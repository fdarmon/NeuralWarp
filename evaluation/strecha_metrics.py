import open3d as o3d
import numpy as np

def strecha_metrics(args):
    scene = args.scene

    in_mesh = o3d.io.read_triangle_mesh(args.in_file)

    stl_pcd_large = o3d.io.read_point_cloud(f"../strecha_eval_zhang/{scene}_gt_ours.ply")
    stl_pcd_centered = o3d.io.read_point_cloud(f"../strecha_eval_zhang/{scene}_gt_centered.ply")

    in_pcd_large = in_mesh.sample_points_uniformly(args.sample, seed=0)

    bb_np = np.load(f"../strecha_eval_zhang/bbox_{scene}.npy")
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bb_np))

    idx_pts = bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(np.asarray(in_mesh.vertices)))
    mask_pts = np.zeros(len(in_mesh.vertices), dtype=bool)
    mask_pts[idx_pts] = True

    np_faces = np.asarray(in_mesh.triangles)

    valid_triangles = mask_pts[np_faces].all(axis=1)
    in_mesh.triangles = o3d.utility.Vector3iVector(np_faces[valid_triangles])
    in_pcd_centered = in_mesh.sample_points_uniformly(args.sample, seed=0)

    thresh_color = [0.8, 0.8, 0.05, 0.05]
    from_l = [in_pcd_large, stl_pcd_large, in_pcd_centered, stl_pcd_centered]
    to_l = [stl_pcd_large, in_pcd_large, stl_pcd_centered, in_pcd_centered]
    names = ["pred2stl", "stl2pred", "pred2stl_centered", "stl2pred_centered"]
    result = {}
    for src, dst, name, thresh_c in zip(from_l, to_l, names, thresh_color):
        res = np.asarray(src.compute_point_cloud_distance(dst))
        print(res.min(), res.mean(), res.max(), res[res < args.thresh].mean())
        result[name] = (res, res[res < args.thresh].mean())

        alpha = res.clip(max=thresh_c) / thresh_c
        #out_of_range = np.expand_dims(res >= args.thresh, 1)
        color = np.stack([np.ones_like(alpha), (1-alpha), (1-alpha)], axis=1)
        #color = color * (~out_of_range) + np.array([[0.0,0.0,1.0]]) * out_of_range
        src.colors = o3d.utility.Vector3dVector(color)
        o3d.io.write_point_cloud(f'{args.evaldir}/{name}.ply', src)

    with open(f'{args.evaldir}/result.txt', 'w') as f:
        f.write(f'{result["pred2stl"][1]} {result["stl2pred"][1]} {(result["pred2stl"][1]+result["stl2pred"][1])/2}\n')
        f.write(
            f'{result["pred2stl_centered"][1]} {result["stl2pred_centered"][1]} {(result["pred2stl_centered"][1] + result["stl2pred_centered"][1]) / 2}')