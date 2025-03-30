import copy

import torch

import PIL.Image as Im
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import numpy.linalg as lg

import yaml
import random
import json
from . import utils, copy_paste_seg
import os
import deep_point
from collections import defaultdict

def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    voxel_feat = deep_point.VoxelMaxPool(pcds_feat=pcds_feat.float(), pcds_ind=pcds_ind, output_size=output_size, scale_rate=scale_rate).to(pcds_feat.dtype)
    return voxel_feat


def make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, Voxel):
    # make point feat
    x = pcds_xyzi[:, 0].copy()
    y = pcds_xyzi[:, 1].copy()
    z = pcds_xyzi[:, 2].copy()
    intensity = pcds_xyzi[:, 3].copy()

    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12

    # grid diff
    diff_x = pcds_coord[:, 0] - np.floor(pcds_coord[:, 0])
    diff_y = pcds_coord[:, 1] - np.floor(pcds_coord[:, 1])
    diff_z = pcds_coord[:, 2] - np.floor(pcds_coord[:, 2])

    # sphere diff
    phi_range_radian = (-np.pi, np.pi)
    theta_range_radian = (Voxel.RV_theta[0] * np.pi / 180.0, Voxel.RV_theta[1] * np.pi / 180.0)

    phi = phi_range_radian[1] - np.arctan2(x, y)
    theta = theta_range_radian[1] - np.arcsin(z / dist)

    diff_phi = pcds_sphere_coord[:, 0] - np.floor(pcds_sphere_coord[:, 0])
    diff_theta = pcds_sphere_coord[:, 1] - np.floor(pcds_sphere_coord[:, 1])

    point_feat = np.stack((x, y, z, intensity, dist, diff_x, diff_y), axis=-1)
    return point_feat


# define the class of dataloader
class DataloadTrain(Dataset):
    def __init__(self, config):
        self.flist = []
        self.config = config
        self.frame_point_num = config.frame_point_num
        self.Voxel = config.Voxel
        with open('datasets/semantic-kitti.yaml', 'r') as f:
            self.task_cfg = yaml.load(f)

        self.cp_aug = None
        if config.CopyPasteAug.is_use:
            self.cp_aug = copy_paste_seg.SequenceCutPaste(config.CopyPasteAug.ObjBackDir, config.CopyPasteAug.paste_max_obj_num)

        self.aug = utils.DataAugmentTemp(noise_mean=config.AugParam.noise_mean,
                        noise_std=config.AugParam.noise_std,
                        theta_range=config.AugParam.theta_range,
                        shift_range=config.AugParam.shift_range,
                        size_range=config.AugParam.size_range)

        self.aug_raw = utils.DataAugmentTemp(noise_mean=0,
                        noise_std=0,
                        theta_range=(0, 0),
                        shift_range=((0, 0), (0, 0), (0, 0)),
                        size_range=(1, 1))

        seq_num = config.seq_num + 2
        # add training data
        self.seq_split = [str(i).rjust(2, '0') for i in self.task_cfg['split']['train']]
        # todo: 只加载包含动态物体的帧
        self.flist = {}
        self.plist = {}
        for seq_id in self.seq_split:
            fpath = os.path.join(config.SeqDir, seq_id)
            fpath_pcd = os.path.join(fpath, 'velodyne')  # velodyne
            fpath_label = os.path.join(fpath, 'labels')  # labels

            fname_calib = os.path.join(fpath, 'calib.txt')  # calib
            fname_pose = os.path.join(fpath, 'poses.txt')  # poses

            calib = utils.parse_calibration(fname_calib)
            poses_list = utils.parse_poses(fname_pose, calib)

            for i in range(len(poses_list)):
                meta_list = []
                meta_list_raw = []
                poses_list_item = []
                current_pose_inv = np.linalg.inv(poses_list[i])
                if (i < (seq_num - 1)):
                    for ht in range(seq_num):
                        file_id = str(i + ht).rjust(6, '0')
                        fname_pcd = os.path.join(fpath_pcd, '{}.bin'.format(file_id))
                        fname_label = os.path.join(fpath_label, '{}.label'.format(file_id))

                        pose_diff = current_pose_inv.dot(poses_list[i + ht])
                        poses_list_item.append(poses_list[i + ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                elif i > (len(poses_list) - seq_num):
                    for ht in range(seq_num):
                        file_id = str(i - ht).rjust(6, '0')
                        fname_pcd = os.path.join(fpath_pcd, '{}.bin'.format(file_id))
                        fname_label = os.path.join(fpath_label, '{}.label'.format(file_id))

                        pose_diff = current_pose_inv.dot(poses_list[i - ht])
                        poses_list_item.append(poses_list[i - ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                else:
                    for ht in range(seq_num):
                        file_id = str(i - ht).rjust(6, '0')
                        fname_pcd = os.path.join(fpath_pcd, '{}.bin'.format(file_id))
                        fname_label = os.path.join(fpath_label, '{}.label'.format(file_id))

                        pose_diff = current_pose_inv.dot(poses_list[i - ht])
                        poses_list_item.append(poses_list[i - ht])
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))

                    for ht in range(seq_num):
                        file_id = str(i + ht).rjust(6, '0')
                        fname_pcd = os.path.join(fpath_pcd, '{}.bin'.format(file_id))
                        fname_label = os.path.join(fpath_label, '{}.label'.format(file_id))

                        pose_diff = current_pose_inv.dot(poses_list[i + ht])
                        poses_list_item.append(poses_list[i + ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))

                if seq_id in self.flist.keys():
                    self.flist[seq_id].append((meta_list, meta_list_raw))
                else:
                    self.flist[seq_id] = [(meta_list, meta_list_raw)]

                if seq_id in self.plist.keys():
                    self.plist[seq_id].append((poses_list_item))
                else:
                    self.plist[seq_id] = [(poses_list_item)]

        print('Training Samples: ', len([item for sublist in self.flist.values() for item in sublist]))

        if config.drop_few_static_frames:
            self.remove_few_static_frames()

        self.flist = [item for sublist in self.flist.values() for item in sublist]
        self.plist = [item for sublist in self.plist.values() for item in sublist]
        print('New Training Samples: ', len(self.flist))

    def form_batch(self, pcds_total, aug_para):
        #augment pcds
        pcds_total = self.aug(pcds_total, aug_para)

        N = pcds_total.shape[0] // self.config.seq_num
        #quantize
        pcds_xyzi = pcds_total[:, :4]
        pcds_coord = utils.Quantize(pcds_xyzi,
                                    range_x=self.Voxel.range_x,
                                    range_y=self.Voxel.range_y,
                                    range_z=self.Voxel.range_z,
                                    size=self.Voxel.bev_shape)

        pcds_sphere_coord = utils.SphereQuantize(pcds_xyzi,
                                            phi_range=(-180.0, 180.0),
                                            theta_range=self.Voxel.RV_theta,
                                            size=self.Voxel.rv_shape)

        #convert numpy matrix to pytorch tensor
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, self.Voxel)  # (x, y, z, intensity, dist, diff_x, diff_y)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32)).view(self.config.seq_num, N, -1, 1)  # [frame_num, point_num, channel, 1]
        pcds_xyzi = pcds_xyzi.permute(0, 2, 1, 3).contiguous()  # [frame_num, channel, point_num, 1]

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)  # [frame_num, point_num, channel, 1]
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)  # [frame_num, point_num, channel, 1]
        return pcds_xyzi, pcds_coord, pcds_sphere_coord

    def form_batch_raw(self, pcds_total, aug_para_raw):
        #augment pcds
        pcds_total = self.aug_raw(pcds_total, aug_para_raw)

        N = pcds_total.shape[0] // self.config.seq_num
        #quantize
        pcds_xyzi = pcds_total[:, :4]
        pcds_coord = utils.Quantize(pcds_xyzi,
                                    range_x=self.Voxel.range_x,
                                    range_y=self.Voxel.range_y,
                                    range_z=self.Voxel.range_z,
                                    size=self.Voxel.bev_shape)

        pcds_sphere_coord = utils.SphereQuantize(pcds_xyzi,
                                            phi_range=(-180.0, 180.0),
                                            theta_range=self.Voxel.RV_theta,
                                            size=self.Voxel.rv_shape)

        #convert numpy matrix to pytorch tensor
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, self.Voxel)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_xyzi = pcds_xyzi.permute(0, 2, 1, 3).contiguous()

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        return pcds_xyzi, pcds_coord, pcds_sphere_coord

    def form_seq(self, meta_list):
        pc_list = []  # point clouds
        pc_label_list = []  # motion-static label
        pc_bf_label_list = []  # motion-static label
        pc_raw_label_list = []
        pc_road_list = []  # road points
        for ht in range(self.config.seq_num + 2):
            fname_pcd, fname_label, pose_diff, _, _ = meta_list[ht]  # points, label, pose
            # load pcd
            pcds_tmp = np.fromfile(fname_pcd, dtype=np.float32).reshape((-1, 4))
            pcds_ht = utils.Trans(pcds_tmp, pose_diff)  # transpose
            pc_list.append(pcds_ht)

            # load label
            pcds_label = np.fromfile(fname_label, dtype=np.uint32)
            pcds_label = pcds_label.reshape((-1))
            sem_label = pcds_label & 0xFFFF
            inst_label = pcds_label >> 16

            pc_road_list.append(pcds_ht[sem_label == 40])

            pcds_label_use = utils.relabel(sem_label, self.task_cfg['learning_map'])
            pcds_bf_label_use = utils.relabel(sem_label, self.task_cfg['bf_learning_map'])
            pc_bf_label_list.append(pcds_bf_label_use)
            pc_label_list.append(pcds_label_use)
            pc_raw_label_list.append(sem_label)

        return pc_list, pc_label_list, pc_bf_label_list, pc_road_list, pc_raw_label_list

    def remove_few_static_frames(self):
        # Developed by Jiadai Sun 2021-11-07
        # This function is used to clear some frames, because too many static frames will lead to a long training time
        # There are several main dicts that need to be modified and processed
        #   self.scan_files, self.label_files, self.residual_files_1 ....8
        #   self.poses, self.index_mapping
        #   self.dataset_size (int number)

        remove_mapping_path = os.path.join(os.path.dirname(__file__),
                                           "../config/train_split_dynamic_pointnumber.txt")
        with open(remove_mapping_path) as fd:
            lines = fd.readlines()
            lines = [line.strip() for line in lines]

        pending_dict = {}
        for line in lines:
            if line != '':
                seq, fid, _ = line.split()
                if seq in self.seq_split:
                    if seq in pending_dict.keys():
                        if fid in pending_dict[seq]:
                            raise ValueError(f"!!!! Duplicate {fid} in seq {seq} in .txt file")
                        pending_dict[seq].append(fid)
                    else:
                        pending_dict[seq] = [fid]

        for seq in self.seq_split:
            # seq = '{0:02d}'.format(int(seq))
            if seq in pending_dict.keys():
                raw_len = len(self.flist[seq])

                # lidar scan files
                scan_files = self.flist[seq]
                useful_scan_index = np.array(pending_dict[seq]).astype('int')
                scan_files = [scan_files[i] for i in useful_scan_index]
                self.flist[seq] = scan_files

                pose_files = self.plist[seq]
                useful_scan_index = np.array(pending_dict[seq]).astype('int')
                pose_files = [pose_files[i] for i in useful_scan_index]
                self.plist[seq] = pose_files

                new_len = len(scan_files)
                print(f"Seq {seq} drop {raw_len - new_len}: {raw_len} -> {new_len}")

    def generate_bev_label(self, pcds_coord, pcds_target):
        pcds_coord_ = torch.clone(pcds_coord[:1, :, :2, :])
        pcds_target_ = torch.clone(pcds_target).unsqueeze(0).unsqueeze(0)
        voxel_size = np.array(self.Voxel.bev_shape[:2]) / 2
        bev_input = VoxelMaxPool(pcds_feat=pcds_target_, pcds_ind=pcds_coord_, output_size=voxel_size.astype('int'), scale_rate=(0.5, 0.5))
        bev_input = bev_input.squeeze(0).squeeze(0).unsqueeze(-1)
        return bev_input

    def __getitem__(self, index):
        batch_dict = []
        aug_para = {}

        meta_list, meta_list_raw = self.flist[index]
        pc_list_all, pc_label_list_all, pc_bf_label_list_all, pc_road_list_all, pc_raw_label_list_all = self.form_seq(meta_list_raw)

        # copy-paste
        if self.cp_aug is not None:
            pc_list_all, pc_label_list_all, pc_bf_label_list_all = self.cp_aug(pc_list_all, pc_label_list_all, pc_bf_label_list_all, pc_road_list_all, pc_raw_label_list_all)
        # draw_single_lidar_with_label(np.concatenate(pc_list_all, axis=0), np.concatenate(pc_label_list_all, axis=0))
        # draw_single_lidar_with_label(np.concatenate(pc_list_all, axis=0), np.concatenate(pc_bf_label_list_all, axis=0))

        ###################################################
        sample_num = 3
        for id in range(len(pc_list_all) - sample_num + 1):
            pc_list = copy.deepcopy(pc_list_all[id: id + (len(pc_list_all) - sample_num) + 1])
            pc_label_list = copy.deepcopy(pc_label_list_all[id: id + (len(pc_list_all) - sample_num) + 1])
            pc_bf_label_list = copy.deepcopy(pc_bf_label_list_all[id: id + (len(pc_list_all) - sample_num) + 1])
            # if id == 0:
                # draw_single_lidar_with_label(np.concatenate(pc_list, axis=0), np.concatenate(pc_label_list, axis=0))
        ###################################################

            ##############
            if id > 0:
                current_pose_inv = np.linalg.inv(self.plist[index][id])
                pose_diff = current_pose_inv.dot(self.plist[index][0])
                for i in range(sample_num):
                    pc_list[i] = utils.Trans(pc_list[i], pose_diff)  # transpose
                # draw_single_lidar_with_label(np.concatenate(pc_list, axis=0), np.concatenate(pc_label_list, axis=0))
            ##############

            # filter
            for ht in range(len(pc_list)):
                valid_mask_ht = utils.filter_pcds_mask(pc_list[ht],
                                                    range_x=self.Voxel.range_x,
                                                    range_y=self.Voxel.range_y,
                                                    range_z=self.Voxel.range_z)

                pc_list[ht] = pc_list[ht][valid_mask_ht]
                pc_label_list[ht] = pc_label_list[ht][valid_mask_ht]
                pc_bf_label_list[ht] = pc_bf_label_list[ht][valid_mask_ht]

            # resample
            for ht in range(len(pc_list)):
                choice = np.random.choice(pc_list[ht].shape[0], self.frame_point_num, replace=True)
                pc_list[ht] = pc_list[ht][choice]
                pc_label_list[ht] = pc_label_list[ht][choice]
                pc_bf_label_list[ht] = pc_bf_label_list[ht][choice]

            pc_list = np.concatenate(pc_list, axis=0)
            pcds_target = torch.LongTensor(pc_label_list[0].astype(np.long)).unsqueeze(-1)
            pcds_bf_target = torch.LongTensor(pc_bf_label_list[0].astype(np.long)).unsqueeze(-1)

            # preprocess
            pcds_xyzi, pcds_coord, pcds_sphere_coord = self.form_batch(pc_list.copy(), aug_para)  # [frame_num, channel, point_num, 1]
            # pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw = self.form_batch_raw(pc_list.copy(), aug_para_raw)

            pcds_bev_target = self.generate_bev_label(pcds_coord, pcds_target)
            # pcds_bev_target_raw = self.generate_bev_label(pcds_coord_raw, pcds_target)

            data_dict = {
                'pcds_xyzi': pcds_xyzi,
                'pcds_coord': pcds_coord,
                'pcds_sphere_coord': pcds_sphere_coord,
                'pcds_bev_target': pcds_bev_target,

                # 'pcds_xyzi_raw': pcds_xyzi_raw,
                # 'pcds_coord_raw': pcds_coord_raw,
                # 'pcds_sphere_coord_raw': pcds_sphere_coord_raw,
                # 'pcds_bev_target_raw': pcds_bev_target_raw,

                'pcds_target': pcds_target,
                'pcds_bf_target': pcds_bf_target,
                'meta_list_raw': meta_list_raw,
            }
            batch_dict.append(data_dict)

        return batch_dict

    def __len__(self):
        return len(self.flist)

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for index, cur_sample_val in enumerate(cur_sample):
                for key, val in cur_sample_val.items():
                    data_dict[key + '_' + str(index)].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['pcds_xyzi_0', 'pcds_coord_0', 'pcds_sphere_coord_0', 'pcds_bev_target_0',
                           'pcds_xyzi_raw_0', 'pcds_coord_raw_0', 'pcds_sphere_coord_raw_0', 'pcds_bev_target_raw_0',
                           'pcds_target_0', 'pcds_bf_target_0',
                           'pcds_xyzi_1', 'pcds_coord_1', 'pcds_sphere_coord_1', 'pcds_bev_target_1',
                           'pcds_xyzi_raw_1', 'pcds_coord_raw_1', 'pcds_sphere_coord_raw_1', 'pcds_bev_target_raw_1',
                           'pcds_target_1', 'pcds_bf_target_1',
                           'pcds_xyzi_2', 'pcds_coord_2', 'pcds_sphere_coord_2', 'pcds_bev_target_2',
                           'pcds_xyzi_raw_2', 'pcds_coord_raw_2', 'pcds_sphere_coord_raw_2', 'pcds_bev_target_raw_2',
                           'pcds_target_2', 'pcds_bf_target_2']:
                    ret[key] = np.stack(val, axis=0)
                elif key in ['box_2d_label', 'box_2d_label_raw']:
                    ret[key] = val
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size

        return ret

# define the class of dataloader
class DataloadVal(Dataset):
    def __init__(self, config):
        self.flist = []
        self.config = config
        self.frame_point_num = config.frame_point_num
        self.Voxel = config.Voxel
        with open('datasets/semantic-kitti.yaml', 'r') as f:
            self.task_cfg = yaml.load(f)

        seq_num = config.seq_num
        # add training data
        seq_split = [str(i).rjust(2, '0') for i in self.task_cfg['split']['valid']]
        for seq_id in seq_split:
            fpath = os.path.join(config.SeqDir, seq_id)
            fpath_pcd = os.path.join(fpath, 'velodyne')
            fpath_label = os.path.join(fpath, 'labels')

            fname_calib = os.path.join(fpath, 'calib.txt')
            fname_pose = os.path.join(fpath, 'poses.txt')

            calib = utils.parse_calibration(fname_calib)
            poses_list = utils.parse_poses(fname_pose, calib)
            for i in range(len(poses_list)):
                meta_list = []
                meta_list_raw = []
                current_pose_inv = np.linalg.inv(poses_list[i])
                if (i < (seq_num - 1)):
                    # backward
                    for ht in range(seq_num):
                        file_id = str(i + ht).rjust(6, '0')
                        fname_pcd = os.path.join(fpath_pcd, '{}.bin'.format(file_id))
                        fname_label = os.path.join(fpath_label, '{}.label'.format(file_id))

                        pose_diff = current_pose_inv.dot(poses_list[i + ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                elif i > (len(poses_list) - seq_num):
                    # forward
                    for ht in range(seq_num):
                        file_id = str(i - ht).rjust(6, '0')
                        fname_pcd = os.path.join(fpath_pcd, '{}.bin'.format(file_id))
                        fname_label = os.path.join(fpath_label, '{}.label'.format(file_id))

                        pose_diff = current_pose_inv.dot(poses_list[i - ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                else:
                    # forward
                    for ht in range(seq_num):
                        file_id = str(i - ht).rjust(6, '0')
                        fname_pcd = os.path.join(fpath_pcd, '{}.bin'.format(file_id))
                        fname_label = os.path.join(fpath_label, '{}.label'.format(file_id))

                        pose_diff = current_pose_inv.dot(poses_list[i - ht])
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))

                    # backward
                    for ht in range(seq_num):
                        file_id = str(i + ht).rjust(6, '0')
                        fname_pcd = os.path.join(fpath_pcd, '{}.bin'.format(file_id))
                        fname_label = os.path.join(fpath_label, '{}.label'.format(file_id))

                        pose_diff = current_pose_inv.dot(poses_list[i + ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))

                self.flist.append((meta_list, meta_list_raw))

        print('Evaluation Samples: ', len(self.flist))

    def form_batch(self, pcds_total):
        N = pcds_total.shape[0] // self.config.seq_num
        # quantize
        pcds_xyzi = pcds_total[:, :4]
        pcds_coord = utils.Quantize(pcds_xyzi,
                                    range_x=self.Voxel.range_x,
                                    range_y=self.Voxel.range_y,
                                    range_z=self.Voxel.range_z,
                                    size=self.Voxel.bev_shape)

        pcds_sphere_coord = utils.SphereQuantize(pcds_xyzi,
                                                 phi_range=(-180.0, 180.0),
                                                 theta_range=self.Voxel.RV_theta,
                                                 size=self.Voxel.rv_shape)

        # convert numpy matrix to pytorch tensor
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, self.Voxel)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_xyzi = pcds_xyzi.permute(0, 2, 1, 3).contiguous()

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        return pcds_xyzi, pcds_coord, pcds_sphere_coord

    def form_batch_tta(self, pcds_total):
        pcds_xyzi_list = []
        pcds_coord_list = []
        pcds_sphere_coord_list = []
        for x_sign in [1, -1]:
            for y_sign in [1, -1]:
                pcds_tmp = pcds_total.copy()
                pcds_tmp[:, 0] *= x_sign
                pcds_tmp[:, 1] *= y_sign
                pcds_xyzi, pcds_coord, pcds_sphere_coord = self.form_batch(pcds_tmp)

                pcds_xyzi_list.append(pcds_xyzi)
                pcds_coord_list.append(pcds_coord)
                pcds_sphere_coord_list.append(pcds_sphere_coord)

        pcds_xyzi = torch.stack(pcds_xyzi_list, dim=0)
        pcds_coord = torch.stack(pcds_coord_list, dim=0)
        pcds_sphere_coord = torch.stack(pcds_sphere_coord_list, dim=0)
        return pcds_xyzi, pcds_coord, pcds_sphere_coord

    def form_seq(self, meta_list):
        pc_list = []
        pc_label_list = []
        for ht in range(self.config.seq_num):
            fname_pcd, fname_label, pose_diff, _, _ = meta_list[ht]
            # load pcd
            pcds_tmp = np.fromfile(fname_pcd, dtype=np.float32).reshape((-1, 4))
            pcds_ht = utils.Trans(pcds_tmp, pose_diff)
            pc_list.append(pcds_ht)

            # load label
            pcds_label = np.fromfile(fname_label, dtype=np.uint32)
            pcds_label = pcds_label.reshape((-1))
            sem_label = pcds_label & 0xFFFF
            inst_label = pcds_label >> 16

            pcds_label_use = utils.relabel(sem_label, self.task_cfg['learning_map'])
            pc_label_list.append(pcds_label_use)

        return pc_list, pc_label_list

    def generate_bev_label(self, pcds_coord, pcds_target):
        pcds_coord_ = torch.clone(pcds_coord[:1, :, :2, :])
        pcds_target_ = torch.clone(pcds_target).unsqueeze(0).unsqueeze(0)
        voxel_size = np.array(self.Voxel.bev_shape[:2]) / 2
        bev_input = VoxelMaxPool(pcds_feat=pcds_target_, pcds_ind=pcds_coord_, output_size=voxel_size.astype('int'), scale_rate=(1.0, 1.0))
        bev_input = bev_input.squeeze(0).squeeze(0).unsqueeze(-1)
        return bev_input

    def __getitem__(self, index):
        meta_list, meta_list_raw = self.flist[index]

        seq_id, file_id = meta_list_raw[0][-2], meta_list_raw[0][-1]

        # load history pcds
        pc_list, pc_label_list = self.form_seq(meta_list_raw)
        pc_raw = pc_list[0]

        valid_mask_list = []
        # filter
        for ht in range(len(pc_list)):
            valid_mask_ht = utils.filter_pcds_mask(pc_list[ht],
                                                   range_x=self.Voxel.range_x,
                                                   range_y=self.Voxel.range_y,
                                                   range_z=self.Voxel.range_z)

            pc_list[ht] = pc_list[ht][valid_mask_ht]
            pc_label_list[ht] = pc_label_list[ht][valid_mask_ht]
            valid_mask_list.append(valid_mask_ht)

        pad_length_list = []
        # max length pad
        for ht in range(len(pc_list)):
            pad_length = self.frame_point_num - pc_list[ht].shape[0]
            assert pad_length > 0
            pc_list[ht] = np.pad(pc_list[ht], ((0, pad_length), (0, 0)), 'constant', constant_values=-1000)
            pc_list[ht][-pad_length:, 2] = -4000

            pc_label_list[ht] = np.pad(pc_label_list[ht], ((0, pad_length),), 'constant', constant_values=0)
            pad_length_list.append(pad_length)

        pc_list = np.concatenate(pc_list, axis=0)
        pcds_target = torch.LongTensor(pc_label_list[0].astype(np.long)).unsqueeze(-1)

        # preprocess
        pcds_xyzi, pcds_coord, pcds_sphere_coord = self.form_batch_tta(pc_list.copy())

        pcds_bev_target = self.generate_bev_label(pcds_coord[0], pcds_target)

        data_dict = {
            'pcds_xyzi': pcds_xyzi,
            'pcds_coord': pcds_coord,
            'pcds_sphere_coord': pcds_sphere_coord,
            'pcds_bev_target': pcds_bev_target,

            'pcds_target': pcds_target,
            'meta_list_raw': meta_list_raw,
            'valid_mask_list': valid_mask_list,
            'pad_length_list': pad_length_list,
            'pc_raw': pc_raw,
            'seq_id': seq_id,
            'file_id': file_id,
        }

        return data_dict

    def __len__(self):
        return len(self.flist)

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['pcds_xyzi', 'pcds_coord', 'pcds_sphere_coord', 'pcds_bev_target',
                           'pcds_xyzi_raw', 'pcds_coord_raw', 'pcds_sphere_coord_raw', 'pcds_bev_target_raw',
                           'pcds_target']:
                    ret[key] = np.stack(val, axis=0)
                elif key in ['box_2d_label', 'box_2d_label_raw']:
                    ret[key] = val
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size

        return ret