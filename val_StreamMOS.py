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
from datetime import datetime

import datasets

from utils.metric import MultiClassMetric
from models import *

import tqdm
import logging
import importlib
from utils.logger import config_logger
from utils import builder
from tensorboardX import SummaryWriter as Logger
# import visualization

#import torch.backends.cudnn as cudnn
#cudnn.deterministic = True
#cudnn.benchmark = False

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

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp

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

    model.eval()
    f = open(os.path.join(save_path, 'record_{}.txt'.format(rank)), 'a')
    query_embed_store = None
    val_save_path = os.path.join(save_path, 'val_results/sequences')
    with torch.no_grad():
        for i, batch_dict in tqdm.tqdm(enumerate(val_loader)):
            load_data_to_gpu(batch_dict)

            # batch_dict['pcds_xyzi'] = batch_dict['pcds_xyzi'][:, :1]
            # batch_dict['pcds_coord'] = batch_dict['pcds_coord'][:, :1]
            # batch_dict['pcds_sphere_coord'] = batch_dict['pcds_sphere_coord'][:, :1]

            pred_cls, pred_res_cls_0, pred_res_cls_1, pred_res_cls_2, query_embed_store = model.infer(batch_dict, i, query_embed_store)

            pred_cls = F.softmax(pred_cls, dim=1)
            pred_cls = pred_cls.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()
            pcds_target = batch_dict['pcds_target'][0, :, 0].contiguous()

            # ################################
            # _, pred_cls = torch.max(pred_cls, dim=1)
            # pcds_xyzi = batch_dict['pcds_xyzi'][0, 0, 0, :3, :, :].transpose(1, 0).cpu().numpy()
            # visualization.draw_single_lidar_with_label(pcds_xyzi, pred_cls)
            # visualization.draw_single_lidar_with_label(pcds_xyzi, pcds_target.cpu().numpy().astype('uint32'))
            ################################

            valid_point_num = pcds_target.shape[0]
            criterion_cate.addBatch(pcds_target, pred_cls[:valid_point_num])

            #################################
            valid_mask = batch_dict['valid_mask_list'][0].reshape(-1)

            new_pred_cls = np.zeros((batch_dict['valid_mask_list'][0].shape[1]))
            _, pred_cls = torch.max(pred_cls, dim=1)
            pred_cls = pred_cls[:pred_cls.shape[0] - batch_dict['pad_length_list'][0][0]]
            new_pred_cls[valid_mask] = pred_cls.cpu().numpy()
            new_pred_cls = new_pred_cls.astype('uint32')

            # ------- save -------
            item_test_save_path = os.path.join(val_save_path, batch_dict['seq_id'][0], 'predictions')
            if not os.path.exists(item_test_save_path):
                os.makedirs(item_test_save_path)

            pred_map = map(new_pred_cls, learning_map_inv)
            pred_map.tofile(os.path.join(item_test_save_path, batch_dict['file_id'][0]+'.label'))

        # record segmentation metric
        metric_cate = criterion_cate.get_metric()
        string = 'Epoch {}'.format(epoch)
        for key in metric_cate:
            string = string + '; ' + key + ': ' + str(metric_cate[key])
            tb_logger.add_scalar(key, metric_cate[key], epoch)

        f.write(string + '\n')
        f.close()
        print(string + '\n')


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel, pOpt = config.get_config()

    prefix = pGen.name  # config
    save_path = os.path.join("experiments", prefix, args.tag)
    model_prefix = os.path.join(save_path, "checkpoint")

    os.system('mkdir -p {}'.format(model_prefix))

    # start logging
    config_logger(os.path.join(save_path, "log_val.txt"))
    logger = logging.getLogger()
    val_tb_logger = Logger(save_path + "/val_tb")

    # reset dist
    device = torch.device('cuda:{}'.format(args.local_rank))
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # reset random seed
    seed = rank * pDataset.Train.num_workers + 50051
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ============== define dataloader ============== !!!!!!!
    val_dataset = eval('datasets.{}.DataloadVal'.format(pDataset.Val.data_src))(pDataset.Val)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=pDataset.Val.num_workers,
                            pin_memory=True)

    print("rank: {}/{}; batch_size: {}".format(rank, world_size, pGen.batch_size_per_gpu))

    for epoch in range(args.start_val_epoch, args.end_val_epoch):
        # ============== define model ============== !!!!!!!
        base_net = eval(pModel.prefix)(pModel)
        # load pretrain model
        pretrain_model = os.path.join(model_prefix, '{}-model.pth'.format(epoch))
        if os.path.exists(pretrain_model):
            base_net.load_state_dict(torch.load(pretrain_model, map_location='cpu'))
            logger.info("Load model from {}".format(pretrain_model))

        base_net = nn.SyncBatchNorm.convert_sync_batchnorm(base_net)
        model = torch.nn.parallel.DistributedDataParallel(base_net.to(device),
                                                        device_ids=[args.local_rank],
                                                        output_device=args.local_rank,
                                                        find_unused_parameters=True)

        # define optimizer
        optimizer = builder.get_optimizer(pOpt, model)

        if rank == 0:
            logger.info(model)
            logger.info(optimizer)

        # start training
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params / 1e6}M")

        val(epoch + rank, base_net, val_loader, pGen.category_list, save_path, val_tb_logger, pDataset.Test.learning_map_inv, rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--tag', help='config file path', type=str, default='base')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--start_val_epoch', type=int, default=40)
    parser.add_argument('--end_val_epoch', type=int, default=41)

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)
