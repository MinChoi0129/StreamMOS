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

#import torch.backends.cudnn as cudnn
#cudnn.deterministic = True
#cudnn.benchmark = False


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
        if not isinstance(val, np.ndarray):
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()

def train(epoch, end_epoch, args, model, train_loader, optimizer, scheduler, logger, tb_logger, log_frequency):
    rank = torch.distributed.get_rank()
    # for id in range(train_loader.dataset.__len__()):
    #     train_loader.dataset.__getitem__(id)
    model.train()
    for i, batch_dict in tqdm.tqdm(enumerate(train_loader)):
        #pdb.set_trace()
        load_data_to_gpu(batch_dict)
        loss = model(batch_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        reduced_loss = reduce_tensor(loss)
        if (i % log_frequency == 0) and rank == 0:
            string = 'Epoch: [{}]/[{}]; Iteration: [{}]/[{}]; lr: {}'.format(epoch, end_epoch,\
                i, len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'])

            string = string + '; loss: {}'.format(reduced_loss.item() / torch.distributed.get_world_size())
            logger.info(string)
            tb_logger.add_scalar('learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], (epoch * len(train_loader) + i))
            tb_logger.add_scalar('loss', reduced_loss.item() / torch.distributed.get_world_size(), (epoch * len(train_loader) + i))


def val(epoch, model, val_loader, category_list, save_path, tb_logger, rank=0):
    criterion_cate = MultiClassMetric(category_list)

    model.eval()
    f = open(os.path.join(save_path, 'record_{}.txt'.format(rank)), 'a')
    query_embed_store = None
    with torch.no_grad():
        for i, batch_dict in tqdm.tqdm(enumerate(val_loader)):
            load_data_to_gpu(batch_dict)
            pred_cls, pred_res_cls_0, pred_res_cls_1, pred_res_cls_2, query_embed_store = model.infer(batch_dict, i, query_embed_store)

            pred_cls = F.softmax(pred_cls, dim=1)
            pred_cls = pred_cls.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()
            pcds_target = batch_dict['pcds_target'][0, :, 0].contiguous()

            valid_point_num = pcds_target.shape[0]
            criterion_cate.addBatch(pcds_target, pred_cls[:valid_point_num])

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
    # save_path = os.path.join("experiments", prefix, args.tag + datetime.now().strftime("-%Y-%-m-%d-%H:%M"))
    save_path = os.path.join("experiments", prefix, args.tag)
    model_prefix = os.path.join(save_path, "checkpoint")

    os.system('mkdir -p {}'.format(model_prefix))

    # start logging
    config_logger(os.path.join(save_path, "log.txt"))
    logger = logging.getLogger()
    train_tb_logger = Logger(save_path + "/train_tb")
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
    train_dataset = eval('datasets.{}.DataloadTrain'.format(pDataset.Train.data_src))(pDataset.Train)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                            batch_size=pGen.batch_size_per_gpu,
                            shuffle=(train_sampler is None),
                            collate_fn=train_dataset.collate_batch,
                            num_workers=pDataset.Train.num_workers,
                            sampler=train_sampler,
                            pin_memory=True)

    val_dataset = eval('datasets.{}.DataloadVal'.format(pDataset.Val.data_src))(pDataset.Val)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=val_dataset.collate_batch,
                            num_workers=pDataset.Val.num_workers,
                            pin_memory=True)

    print("rank: {}/{}; batch_size: {}".format(rank, world_size, pGen.batch_size_per_gpu))

    # ============== define model ============== !!!!!!!
    base_net = eval(pModel.prefix)(pModel)
    # load pretrain model
    pretrain_model = os.path.join(model_prefix, '{}-model.pth'.format(pModel.pretrain.pretrain_epoch))
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

    # define scheduler
    per_epoch_num_iters = len(train_loader)
    scheduler = builder.get_scheduler(optimizer, pOpt, per_epoch_num_iters)

    if rank == 0:
        logger.info(model)
        logger.info(optimizer)
        logger.info(scheduler)

    # start training
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6}M")
    for epoch in range(pOpt.schedule.begin_epoch, pOpt.schedule.end_epoch):
        train_sampler.set_epoch(epoch)
        train(epoch, pOpt.schedule.end_epoch, args, model, train_loader, optimizer, scheduler, logger, train_tb_logger, pGen.log_frequency)

        # save model
        if rank == 0:
            torch.save(model.module.state_dict(), os.path.join(model_prefix, '{}-model.pth'.format(epoch)))

        if epoch >= args.start_val_epoch:
            val(epoch + rank, base_net, val_loader, pGen.category_list, save_path, val_tb_logger, rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--tag', help='config file path', type=str, default='base')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--start_val_epoch', type=int, default=40)

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)
