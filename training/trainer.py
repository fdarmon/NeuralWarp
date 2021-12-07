import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
import time
from model.loss import Loss
from model.neuralWarp import NeuralWarp
from pathlib import Path
from datasets.scene_dataset import SceneDataset

import utils.general as utils
import utils.plots as plt

from utils import html_writer


from tqdm import tqdm

torch.backends.cudnn.benchmark = False

class Trainer():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.debug = kwargs["debug"]

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.expname = Path(kwargs["conf"]).with_suffix("").name + kwargs['expname']
        self.exps_folder_name = Path("exps")
        if len(Path(kwargs["conf"]).parents) == 3:
            self.exps_folder_name = self.exps_folder_name / Path(kwargs["conf"]).parents[0].name
        elif len(Path(kwargs["conf"]).parents) > 3:
            print("Too many depth levels in directory structure")
            raise RuntimeError

        scan_id = kwargs['scan_id'] if kwargs['scan_id'] is not None else self.conf.get_int('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        self.niterations = self.conf.get_int('train.niterations')
        self.finetune_exp = self.conf.get_string("train.finetune_exp", None)
        if self.finetune_exp is not None and not kwargs['is_continue']:
            self.exp_dir_load = Path("exps") / (self.finetune_exp + '_{0}'.format(scan_id))
        else:
            self.exp_dir_load = self.exps_folder_name / self.expname

        if self.finetune_exp is not None or (kwargs['is_continue'] and kwargs['timestamp'] == 'latest'):
            if self.exp_dir_load.exists():
                timestamps = list(self.exp_dir_load.iterdir())
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1].name
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        self.expdir = self.exps_folder_name / self.expname
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        self.plots_dir = self.expdir / self.timestamp / 'plots'
        self.checkpoints_path = self.expdir / self.timestamp / 'checkpoints'
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        if not self.debug:
            # seems to solve the problem with jz
            utils.mkdir_ifnotexists(self.exps_folder_name)
            utils.mkdir_ifnotexists(self.expdir)
            utils.mkdir_ifnotexists(self.expdir / self.timestamp)
            utils.mkdir_ifnotexists(self.plots_dir)
            utils.mkdir_ifnotexists(self.checkpoints_path)
            utils.mkdir_ifnotexists(self.checkpoints_path / self.model_params_subdir)
            utils.mkdir_ifnotexists(self.checkpoints_path / self.optimizer_params_subdir)
            utils.mkdir_ifnotexists(self.checkpoints_path / self.scheduler_params_subdir)

            os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], str(self.expdir / self.timestamp / 'runconf.conf')))

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        dataset_conf["nsrc"] = self.conf["train.nviews"]
        dataset_conf["uv_down"] = self.conf.get_int("plot.uv_down")
        self.patch_loss = self.conf.get_bool("train.patch_loss")
        self.h_patch_size = self.conf.get_int("train.half_patch_size")

        self.train_dataset = SceneDataset(h_patch_size=self.h_patch_size if self.patch_loss else None,
                                          **dataset_conf)

        self.train_imgs = self.train_dataset.rgb_images.cuda()

        print('Finish loading data ...')

        self.model = NeuralWarp(conf=self.conf.get_config('model'))

        if torch.cuda.is_available():
            self.model.cuda()
            self.device = torch.device("cuda")

        # manually set some model args
        self.model.warped_colors = self.conf.get_float("loss.warping_weight", 0) > 0
        self.model.network_colors = self.conf.get_float("loss.network_weight", 0) > 0
        if self.patch_loss:
            self.model.update_patch_size(self.h_patch_size)

        self.loss = Loss(**self.conf.get_config('loss'), h_patch_size=self.h_patch_size)
        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if "finetune" in self.conf.train:
            self.finetune_conf = self.conf.train.finetune
            self.finetuning_iteration = self.finetune_conf.iteration
        else:
            self.finetuning_iteration = float("inf")

        sched_conf = self.conf.get_config("train.scheduler", None)
        if sched_conf is None:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [], gamma=1)
        elif sched_conf["type"] == "step":
            sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
            sched_factor = self.conf.get_float('train.sched_factor', default=1.0)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, sched_milestones, gamma=sched_factor)
        else:
            niter = self.niterations if not "finetune" in self.conf.train else self.finetuning_iteration
            end_factor = sched_conf.get_float("end_factor")
            gamma = end_factor ** (1 / niter)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        self.start_iteration = 0
        if is_continue:
            old_checkpnts_dir = self.exp_dir_load / timestamp / 'checkpoints'

            saved_model_state = torch.load((old_checkpnts_dir / 'ModelParameters' / str(kwargs['checkpoint'])).with_suffix(".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"], strict=False)
            if self.model.occlusion_detector is not None:
                self.model.occlusion_detector.implicit_network = self.model.implicit_network

            self.start_iteration = saved_model_state['iteration'] if (self.finetune_exp is None or kwargs["is_continue"]) else 0

            data = torch.load((old_checkpnts_dir / 'OptimizerParameters' / str(kwargs['checkpoint'])).with_suffix(".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load((old_checkpnts_dir / self.scheduler_params_subdir / str(kwargs['checkpoint'])).with_suffix(".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

        if self.debug:
            self.niterations = min(self.niterations, self.start_iteration + 30)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            pin_memory=True
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn,
                                                           )

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.plot_img_res
        self.plot_pixels = self.img_res[0] * self.img_res[1]
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('plot.plot_freq')
        self.save_freq = self.conf.get_int("train.save_freq")
        self.plot_conf = self.conf.get_config('plot')

        self.running_avg = {}

        self.per_image_loss = {}

        self.ft_config_changed = False


    def log_losses(self, iteration):
        with open(self.checkpoints_path.parent / "logs.txt", "a") as f:
            s = f"Iteration: {iteration}"
            for k in self.running_avg:
                s += f" {k}: {self.running_avg[k]}"
            f.write(s)
            f.write("\n")

        self.running_avg = {}

    def save_checkpoints(self, iteration):
        torch.save(
            {"iteration": iteration, "model_state_dict": self.model.state_dict()},
            (self.checkpoints_path / self.model_params_subdir / str(iteration)).with_suffix(".pth"))
        torch.save(
            {"iteration": iteration, "model_state_dict": self.model.state_dict()},
            self.checkpoints_path / self.model_params_subdir / "latest.pth")

        torch.save(
            {"iteration": iteration, "optimizer_state_dict": self.optimizer.state_dict()},
            (self.checkpoints_path / self.optimizer_params_subdir / str(iteration)).with_suffix(".pth"))
        torch.save(
            {"iteration": iteration, "optimizer_state_dict": self.optimizer.state_dict()},
            self.checkpoints_path / self.optimizer_params_subdir / "latest.pth")

        torch.save(
            {"iteration": iteration, "scheduler_state_dict": self.scheduler.state_dict()},
            (self.checkpoints_path / self.scheduler_params_subdir / str(iteration)).with_suffix(".pth"))
        torch.save(
            {"iteration": iteration, "scheduler_state_dict": self.scheduler.state_dict()},
            self.checkpoints_path / self.scheduler_params_subdir / "latest.pth")

    def plot(self, iteration):
        self.model.render_normals = True
        self.model.eval()
        self.train_dataset.change_sampling_idx(-1)
        plot_indices, plot_model_input = next(iter(self.plot_dataloader))

        plot_model_input["intrinsics"] = plot_model_input["intrinsics"].cuda()
        plot_model_input["uv"] = plot_model_input["uv"].cuda()
        plot_model_input['pose'] = plot_model_input['pose'].cuda()
        plot_model_input["inverse_pose"] = plot_model_input['inverse_pose'].cuda()
        plot_model_input["inverse_intrinsics"] = plot_model_input["inverse_intrinsics"].cuda()
        plot_model_input["rgb"] = self.train_imgs[plot_model_input["idx_list"].squeeze(0)].unsqueeze(0)

        split = utils.split_input(plot_model_input, self.plot_pixels, 500)
        res = []
        self.model.update_patch_size(None)
        for s in tqdm(split, desc="generate_visu"):
            out = self.model(s)
            utils.detach(out)
            utils.to(out, torch.device("cpu"))
            res.append(out)

        utils.to(plot_model_input, torch.device("cpu"))
        batch_size = plot_model_input['rgb'].shape[0]
        model_output = utils.merge_output(res, self.plot_pixels, batch_size)

        plt.plot(self.model,
                 plot_model_input,
                 model_output,
                 self.plots_dir,
                 iteration,
                 self.img_res,
                 **self.plot_conf
                 )

        html_writer.pack_folder_to_html(self.plots_dir)

        # remove all intermediate results, there can be oom after a visualization step
        del split, plot_model_input, model_output, res, out, s

        self.model.train()
        self.model.render_normals = False
        self.train_dataset.change_sampling_idx(self.num_pixels)
        if self.patch_loss:
            self.model.update_patch_size(self.h_patch_size)

    def run(self):
        iteration = self.start_iteration
        while True:
            self.train_dataset.change_sampling_idx(self.num_pixels)

            for data_index, (indices, model_input) in enumerate(self.train_dataloader):
                if not self.ft_config_changed and iteration >= self.finetuning_iteration:  # update parameters for the finetuning iterations
                    self.patch_loss = self.finetune_conf.patch_loss
                    self.num_pixels = self.finetune_conf.num_pixels
                    self.loss = Loss(**self.finetune_conf.get_config('loss'), h_patch_size=self.h_patch_size)

                    # update model and dataset attributes accordingly
                    if self.patch_loss:
                        self.model.update_patch_size(self.h_patch_size)
                        self.train_dataset.h_patch_size = self.h_patch_size
                    self.model.warped_colors = self.finetune_conf.get_float("loss.warping_weight", 0) > 0
                    self.model.network_colors = self.finetune_conf.get_float("loss.network_weight", 0) > 0

                    self.ft_config_changed = True
                    break # break so that we start from a new dataloader

                if self.save_freq > -1 and not self.debug and iteration % self.save_freq == 0:
                    self.save_checkpoints(iteration)

                if self.plot_freq > -1 and not self.debug and iteration % self.plot_freq == 0:
                    torch.cuda.empty_cache()
                    self.plot(iteration)
                    torch.cuda.empty_cache()

                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                uv_all = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                model_input["inverse_pose"] = model_input['inverse_pose'].cuda()
                model_input["inverse_intrinsics"] = model_input["inverse_intrinsics"].cuda()

                model_input["rgb"] = self.train_imgs[model_input["idx_list"].squeeze(0)].unsqueeze(0)

                self.optimizer.zero_grad()

                model_input["uv"] = uv_all

                model_output = self.model(model_input)
                loss_output = self.loss(model_input, model_output)

                loss = loss_output['loss']
                if torch.isnan(loss):
                    print("nan")
                    raise RuntimeError("Nan in loss")

                for l in loss_output:
                    self.running_avg[l] = self.running_avg.get(l, 0) + loss_output[l].item() / len(self.train_dataloader)

                loss.backward()
                self.optimizer.step()

                str = f"{self.expname} [{iteration}]: loss = {loss_output['loss'].item():.3f}"
                for k in loss_output:
                    if k != "loss":
                        str += f", {k} = {loss_output[k].item():.3f}"
                str += f" Beta: {self.model.implicit_network.beta().item():.3f}"

                print(str)

                self.scheduler.step()
                iteration += 1
                if iteration >= self.niterations:
                    break

            if not self.debug:
                self.log_losses(iteration)

            if iteration >= self.niterations:
                break
