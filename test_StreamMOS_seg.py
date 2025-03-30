import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pdb

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets

from utils.metric import MultiClassMetric
from models import *
# from visualization import draw_single_lidar_with_label
import tqdm
import importlib
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter as Logger
cudnn.benchmark = True
cudnn.enabled = True

def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
        if isinstance(data, list):
            nel = len(data)
        else:
            nel = 1
        if key > maxkey:
            maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
        lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
        lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
        try:
            lut[key] = data
        except IndexError:
            print("Wrong key ", key)
    # do the mapping
    return lut[label]

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if key in ['box_2d_label', 'box_2d_label_raw']:
            for index, this_item in enumerate(val):
                val[index]['boxes'] = torch.from_numpy(val[index]['boxes']).float().cuda()
                val[index]['labels'] = torch.from_numpy(val[index]['labels']).long().cuda()
        if isinstance(val, list):
            continue
        batch_dict[key] = val.float().cuda()

def val(epoch, model, val_loader, category_list, save_path, tb_logger, learning_map_inv, rank=0):
    criterion_cate = MultiClassMetric(category_list)
    # for id in range(val_loader.dataset.__len__()):
    #     val_loader.dataset.__getitem__(id)
    model.eval()
    f = open(os.path.join(save_path, 'record_{}.txt'.format(rank)), 'a')
    query_embed_store = None
    test_save_path = 'test/sequences'
    test_bf_save_path = 'test_bf/sequences'
    with torch.no_grad():
        for i, batch_dict in tqdm.tqdm(enumerate(val_loader)):
            load_data_to_gpu(batch_dict)
            pred_cls, bf_pred_cls, pred_res_cls_0, pred_res_cls_1, pred_res_cls_2, query_embed_store = model.infer(batch_dict, i, query_embed_store)

            pred_cls = F.softmax(pred_cls, dim=1)
            pred_cls = pred_cls.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()

            bf_pred_cls = F.softmax(bf_pred_cls, dim=1)
            bf_pred_cls = bf_pred_cls.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()
            _, bf_pred_cls = bf_pred_cls.max(dim=1)

            # ------- vis & reshape data !!!! -------
            valid_mask = batch_dict['valid_mask_list'][0].reshape(-1)

            new_pred_cls = np.zeros((batch_dict['valid_mask_list'][0].shape[1]))
            _, pred_cls = torch.max(pred_cls, dim=1)
            pred_cls = pred_cls[:pred_cls.shape[0] - batch_dict['pad_length_list'][0][0]]
            new_pred_cls[valid_mask] = pred_cls.cpu().numpy()
            new_pred_cls = new_pred_cls.astype('uint32')

            new_bf_pred_cls = np.zeros((batch_dict['valid_mask_list'][0].shape[1]))
            bf_pred_cls = bf_pred_cls[:bf_pred_cls.shape[0] - batch_dict['pad_length_list'][0][0]]
            new_bf_pred_cls[valid_mask] = bf_pred_cls.cpu().numpy()
            new_bf_pred_cls = new_bf_pred_cls.astype('uint32')

            # ------- save -------
            item_test_save_path = os.path.join(test_save_path, batch_dict['seq_id'][0], 'predictions')
            item_test_bf_save_path = os.path.join(test_bf_save_path, batch_dict['seq_id'][0], 'predictions')
            if not os.path.exists(item_test_save_path):
                os.makedirs(item_test_save_path)
            if not os.path.exists(item_test_bf_save_path):
                os.makedirs(item_test_bf_save_path)

            pred_map = map(new_pred_cls, learning_map_inv)
            pred_map.tofile(os.path.join(item_test_save_path, batch_dict['file_id'][0]+'.label'))
            new_bf_pred_cls.tofile(os.path.join(item_test_bf_save_path, batch_dict['file_id'][0] + '.label'))


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel, pOpt = config.get_config()
    
    prefix = pGen.name
    save_path = os.path.join("experiments", prefix, args.tag)
    model_prefix = os.path.join(save_path, "checkpoint")
    tb_logger = Logger(save_path + "/test_tb")

    # reset dist
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # define dataloader
    val_dataset = eval('datasets.{}.DataloadTest'.format(pDataset.Test.data_src))(pDataset.Test)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=pDataset.Val.num_workers,
                            pin_memory=True)
    
    # define model
    model = eval(pModel.prefix)(pModel)
    model.cuda()
    model.eval()
    
    for epoch in range(args.start_epoch, args.end_epoch + 1, world_size):
        if (epoch + rank) < (args.end_epoch + 1):
            pretrain_model = os.path.join(model_prefix, '{}-model.pth'.format(epoch + rank))
            model.load_state_dict(torch.load(pretrain_model, map_location='cpu'))
            val(epoch + rank, model, val_loader, pGen.category_list, save_path, tb_logger, pDataset.Test.learning_map_inv, rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tag', help='config file path', type=str, default='base')
    parser.add_argument('--start_epoch', type=int, default=10)
    parser.add_argument('--end_epoch', type=int, default=11)
    
    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)