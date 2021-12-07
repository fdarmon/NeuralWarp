import imageio
import numpy as np
import plotly.graph_objs as go
import plotly.offline as offline
import torch
import trimesh
from PIL import Image
from matplotlib import pyplot as plt
from skimage import measure
from torch.nn import functional as F
from torchvision.transforms import ToPILImage

from utils import rend_util
from utils import utils_3D


def plot(model, model_inputs ,model_outputs, path, iteration, img_res, max_depth, resolution, **kwargs):
    # arrange data to plot

    pose = model_inputs["pose"][:, 0]
    batch_size, num_samples, _ = model_inputs["uv"].shape
    rgb_gt = model_inputs["rgb"]
    _, nimgs, _, h, w = rgb_gt.shape

    network_object_mask = model_outputs["network_object_mask"]
    points = model_outputs["intersection_points"].reshape(batch_size, num_samples, 3)

    depth = torch.ones(batch_size * num_samples).float() * max_depth
    depth[network_object_mask] = rend_util.get_depth(points, pose).reshape(-1)[network_object_mask]
    depth = depth.reshape(batch_size, num_samples, 1)

    # plot gt
    for i in range(nimgs):
        save_gt(rgb_gt[0, i], path, iteration, i)

    grid = utils_3D.normalize(model_inputs["uv"], h, w).unsqueeze(1)
    rgb_gt_aligned = F.grid_sample(rgb_gt[:, 0], grid, align_corners=True).squeeze().t().reshape(batch_size, num_samples, 3)

    # network renderings
    if "network_rgb_values" in model_outputs:
        network_rgb_eval = model_outputs["network_rgb_values"]
        network_rgb_eval = network_rgb_eval.reshape(batch_size, num_samples, 1, 3)
        plot_images(network_rgb_eval, rgb_gt_aligned, path, iteration, img_res, f"network")

    # warping renderings
    if "warped_rgb_values" in model_outputs:
        warp_rgb_eval = model_outputs["warped_rgb_values"]
        nsrc = warp_rgb_eval.shape[1]
        warp_rgb_eval = warp_rgb_eval.reshape(batch_size, num_samples, nsrc, 3)
        warp_rgb_eval = model_outputs[f"warping_mask"].unsqueeze(-1) * warp_rgb_eval + (1 - model_outputs[f"warping_mask"]).unsqueeze(-1) * torch.tensor([0, 1, 0]).float()

        if f"occlusion_mask" in model_outputs:
            occmask = model_outputs[f"occlusion_mask"]
            inv_occmask = 1 - occmask
            warp_rgb_eval = inv_occmask.unsqueeze(-1) * warp_rgb_eval +  occmask.unsqueeze(-1) * torch.tensor([0, 0, 1]).float()
        plot_images(warp_rgb_eval, rgb_gt_aligned, path, iteration, img_res, f"warping")

    # plot depth maps
    plot_depth_maps(depth, path, iteration, img_res)

    # plot normal maps
    if f"avg_normals" in model_outputs and model_outputs["avg_normals"] is not None:
        normals = model_outputs[f"avg_normals"].reshape(batch_size, num_samples, 3)
        plot_normal_maps(normals, path, iteration, img_res)

    network_object_mask = model_outputs["network_object_mask"].reshape(batch_size, -1)
    sampled_points = model_outputs["sampled_points"].reshape(batch_size, num_samples, -1, 3)
    cam_loc, cam_dir = rend_util.get_camera_for_plot(pose)

    data = []

    # plot surface
    fun = lambda x: model.implicit_network(x)[:, 0]
    surface_traces = get_surface_trace(path=path,
                                       iteration=iteration,
                                       occ=fun,
                                       resolution=resolution
                                       )

    if surface_traces is None:
        return

    data.append(surface_traces[0])

    # plot cameras locations
    for loc, dir in zip(cam_loc, cam_dir):
        data.append(get_3D_quiver_trace(loc.unsqueeze(0), - dir.unsqueeze(0), name=f"camera_{i}"))

    # plot points intersection
    for s, m in zip(sampled_points, network_object_mask):
        s = s[m]
        # v = v[m]
        sampling_idx = torch.randperm(s.shape[0])[:40]
        s = s[sampling_idx, :]
        #v = v[sampling_idx, :]
        N, Ns, _ = s.shape

        t1, t2 = get_3D_scatter_trace(s.view(-1, 3))
        data.append(t1)
        data.append(t2)

    fig = go.Figure(data=data)
    scene_dict = dict(xaxis=dict(range=[-3, 3], autorange=False),
                      yaxis=dict(range=[-3, 3], autorange=False),
                      zaxis=dict(range=[-3, 3], autorange=False),
                      aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=scene_dict, width=1400, height=1400, scene_dragmode="orbit", showlegend=True)
    filename = f"{path}/surface_{iteration}.html"
    offline.plot(fig, filename=filename, auto_open=False)


def get_3D_scatter_trace(points, name="", size=2, visible=None, value=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    if value is None:
        value = torch.ones_like(points[..., 0])
    if visible is None:
        visible = torch.ones_like(value, dtype=torch.bool)
    trace1 = go.Scatter3d(
        x=points[visible, 0].cpu(),
        y=points[visible, 1].cpu(),
        z=points[visible, 2].cpu(),
        mode="markers",
        name=name,
        marker=dict(
            size=size,
            color=value[visible].detach().cpu(),
            opacity=1,
            line=dict(
                width=2,
            ),
        ))
    trace2 = go.Scatter3d(
        x=points[~visible, 0].cpu(),
        y=points[~visible, 1].cpu(),
        z=points[~visible, 2].cpu(),
        mode="markers",
        name=name,
        marker=dict(
            size=size,
            color="#00ff00",
            opacity=1,
            line=dict(
                width=2,
            ),
        ))

    return trace1, trace2


def get_3D_quiver_trace(points, directions, color='#bd1540', name=''):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "

    trace = go.Cone(
        name=name,
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        u=directions[:, 0].cpu(),
        v=directions[:, 1].cpu(),
        w=directions[:, 2].cpu(),
        sizemode='absolute',
        sizeref=0.125,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tip"
    )

    return trace


def get_surface_trace(path, iteration, occ, resolution=100, return_mesh=False):
    grid = get_grid_uniform(resolution, bbox_size=1.)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(occ(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > 0 or np.max(z) < 0)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=0,
            gradient_direction="ascent",
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface', hoverinfo="skip",
                            opacity=1)]

        meshexport = trimesh.Trimesh(verts, faces, vertex_normals=normals)
        meshexport.export('{0}/surface_{1}.ply'.format(path, iteration), 'ply')

        if return_mesh:
            return meshexport

        return traces
    return None

def get_grid_uniform(resolution, bbox_size):
    x = np.linspace(-bbox_size, bbox_size, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points.cuda(),
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

def get_grid(points, resolution):
    eps = 0.2
    input_min = torch.min(points, dim=0)[0].squeeze().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}

def plot_depth_maps(depth_maps, path, iteration, img_res):
    depth_maps_plot = lin2img(depth_maps.cpu(), img_res)
    depth_maps_plot = np.tile(depth_maps_plot, (1, 1, 1, 3))
    img = Image.fromarray((255 * depth_maps_plot.squeeze(0) / depth_maps_plot.max()).astype(np.uint8))
    img.save(f"{path}/depth_{iteration}.jpg")

def plot_normal_maps(normal_maps, path, iteration, img_res):
    normal_maps_plot = lin2img(normal_maps.cpu(), img_res)
    img = Image.fromarray((255 * ((normal_maps_plot+1)/2).squeeze(0)).astype(np.uint8))
    img.save(f"{path}/normal_{iteration}.jpg")

def save_gt(ground_true, path, iteration, idx_view):
    pilimage = ToPILImage()(ground_true.cpu())
    w, h  = pilimage.size
    ratio = max(w / 400, h / 400)
    pilimage = pilimage.resize((int(w / ratio), int(h / ratio)), Image.LANCZOS if ratio < 1 else Image.BILINEAR)
    pilimage.save(f"{path}/gt_view{idx_view}_{iteration}.jpg")

def plot_images(rgb_points, ground_true, path, iteration, img_res, name):
    ground_true = ground_true
    rgb_points = rgb_points

    nsrc = rgb_points.shape[2]
    img_list = [rgb_points[:, :, i] for i in range(nsrc)]
    img_list = [ground_true, ] + img_list

    output_list = [lin2img(torch.clamp(255 * img, 0, 255).long().cpu(), img_res).squeeze(0).astype(np.uint8) for img in img_list]

    imageio.mimsave('{0}/{1}_rendering_{2}.gif'.format(path, name, iteration), output_list, duration=1)

def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.view(batch_size, img_res[0], img_res[1], channels).numpy()
