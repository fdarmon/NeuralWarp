#  NeuralWarp  All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#    * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#    * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

import os
from glob import glob
import torch

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def glob_imgs(path):
    if path is None:
        return []
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def split_input(model_input, total_pixels, n_pixels=10000):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        split.append(data)
    return split

def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue

        to_cat = [r[entry] for r in res]
        if len(to_cat[0].shape) == 0:
            model_outputs[entry] = torch.stack(to_cat, dim=0)
        else:
            model_outputs[entry] = torch.cat(to_cat, dim=0)
        # if len(res[0][entry].shape) == 1:
        #     model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
        #                                      1).reshape(batch_size * total_pixels)
        # else:
        #     model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
        #                                      1).reshape(batch_size * total_pixels, -1)

    return model_outputs

def to(batch, device=torch.device("cuda")):
    for b in batch:
        if isinstance(batch[b], torch.Tensor):
            batch[b] = batch[b].to(device)

def detach(batch):
    for b in batch:
        if isinstance(batch[b], torch.Tensor):
            batch[b] = batch[b].detach()

def print_memory_usage(prefix=""):
    usage = {}
    for attr in ["memory_allocated", "max_memory_allocated", "memory_reserved", "max_memory_reserved"]:
        usage[attr] = getattr(torch.cuda, attr)() * 0.000001
    print("{}:\t{}".format(
        prefix, " / ".join(["{}: {:.0f}MiB".format(k, v) for k, v in usage.items()])))

