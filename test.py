import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from model import StainPresetNet


def list_file_tree(path, file_type="tif"):
    image_list = list()
    dir_list = os.listdir(path)
    if os.path.isdir(path):
        image_list += glob(os.path.join(path, "*" + file_type))
    for dir_name in dir_list:
        sub_path = os.path.join(path, dir_name)
        if os.path.isdir(sub_path):
            image_list += list_file_tree(sub_path, file_type)
    return image_list


def norm(image):
    image = np.array(image).astype(np.float32)
    image = image.transpose((2, 0, 1))
    image = ((image / 255) - 0.5) / 0.5
    image = image[np.newaxis, ...]
    image = torch.from_numpy(image)
    return image


def un_norm(image):
    image = image.cpu().detach().numpy()[0]
    image = ((image * 0.5 + 0.5) * 255).astype(np.uint8).transpose((1, 2, 0))
    return image


def run_norm(model_path, root_path, save_root):
    source_files = list_file_tree(root_path, 'png')
    model = StainPresetNet().cuda()
    model.load_state_dict(torch.load(model_path)['net_G_A'])
    for ref_path in source_files:
        ref = Image.open(ref_path)
        model.set_ref(norm(ref).cuda())
        save_path = os.path.join(save_root, os.path.split(ref_path)[1][:-4])
        for bb in tqdm(source_files):
            dir_root, filename = os.path.split(bb)
            save_dir = dir_root.replace(root_path, save_path)
            os.makedirs(save_dir, exist_ok=True)
            img = model.style(norm(Image.open(bb).convert('RGB')).cuda())
            img = un_norm(img)
            Image.fromarray(img).save(os.path.join(save_dir, filename))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    run_norm("checkpoints/StainPresetGAN-Hist-camelyon16&17.pt",
             "./imgs",
             "./result")
