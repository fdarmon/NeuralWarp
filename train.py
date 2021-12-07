import argparse

import torch
import numpy as np

from training.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--conf', type=str, default='./confs/NeuralWarp.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint iteration number of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--scan_id', default=None, help='If set, taken to be the scan id.')
    parser.add_argument('--exps_folder_name', type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=1995)
    opt = parser.parse_args()

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    trainrunner = Trainer(conf=opt.conf,
                          batch_size=opt.batch_size,
                          expname=opt.expname,
                          exps_folder_name=opt.exps_folder_name,
                          is_continue=opt.is_continue,
                          timestamp=opt.timestamp,
                          checkpoint=opt.checkpoint,
                          scan_id=opt.scan_id,
                          debug=opt.debug,
                          )

    trainrunner.run()
