import copy
import numpy as np
import torch
import utils.transforms as tr
import os
import yaml
import datasets.utils as utils
import tqdm
import argparse
import importlib
from utils.metric import MultiClassMetric

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

def get_point_labels_from_voxel_labels(new_voxel_coords, voxel_labels, size, scale=[1,1]):
    scale_xy = scale[0]
    scale_z = scale[1]

    #********
    x_max, y_max, z_max = size[0] // scale_xy, size[1] // scale_xy, size[2] // scale_z

    valid_mask = (new_voxel_coords >= 0).all(dim=1) & (new_voxel_coords[:, 0] < x_max) & (new_voxel_coords[:, 1] < y_max) & (new_voxel_coords[:, 2] < z_max) 

    point_labels = torch.zeros_like(valid_mask, dtype=torch.long, device=new_voxel_coords.device)

    valid_voxel_coords = new_voxel_coords[valid_mask]
    linear_indices = valid_voxel_coords[:, 0] * voxel_labels.size(1) * voxel_labels.size(2) + valid_voxel_coords[:, 1] * voxel_labels.size(2) + valid_voxel_coords[:, 2]
    point_labels[valid_mask] = voxel_labels.view(-1)[linear_indices]
    
    return point_labels

def determine_voxel_labels(voxel_coords, semantic_labels, size, scale=[1, 1]):
    scale_xy = scale[0]
    scale_z = scale[1]
    # assert (semantic_labels >= 0).all(), "Negative values found in semantic_labels"
    num_classes = semantic_labels.max().item() + 1

    x_max, y_max, z_max = size[0] // scale_xy, size[1] // scale_xy, size[2] // scale_z

    # nonzero_mask =  (voxel_coords.min(dim=1)[0] >= 0) & (voxel_coords[:, 0] < x_max) & (voxel_coords[:, 1] < y_max) & (voxel_coords[:, 2] < z_max)
    # voxel_coords = voxel_coords[nonzero_mask]
    # semantic_labels = semantic_labels[nonzero_mask]

    linear_indices = voxel_coords[:, 0] * y_max * z_max + voxel_coords[:, 1] * z_max + voxel_coords[:, 2]

    votes = torch.zeros(x_max * y_max * z_max, num_classes, dtype=torch.long, device=voxel_coords.device)

    votes.scatter_add_(0, linear_indices.unsqueeze(1).expand(-1, num_classes), torch.nn.functional.one_hot(semantic_labels, num_classes))

    voxel_labels = votes.view(x_max, y_max, z_max, num_classes).argmax(dim=-1)

    return voxel_labels

def Quantize(pcds, range_x=(-40, 62.4), range_y=(-40, 40), range_z=(-3, 5), size=(512, 512, 20)):
    size_x = size[0]
    size_y = size[1]
    size_z = size[2]
    
    dx = (range_x[1] - range_x[0]) / size_x
    dy = (range_y[1] - range_y[0]) / size_y
    dz = (range_z[1] - range_z[0]) / size_z
    
    x_quan = ((pcds[:, 0] - range_x[0]) / dx)
    y_quan = ((pcds[:, 1] - range_y[0]) / dy)
    z_quan = ((pcds[:, 2] - range_z[0]) / dz)
    
    pcds_quan = torch.stack((x_quan, y_quan, z_quan), dim=-1)
    return pcds_quan

def get_data(data_path, pred_path, file_name):
    points = np.fromfile(data_path + file_name, dtype=np.float32).reshape((-1, 4))
    pcds_label = np.fromfile(pred_path + file_name.split('.')[0] + '.label', dtype=np.uint32)
    sem_label = pcds_label & 0xFFFF
    pred_result = utils.relabel(sem_label, task_cfg['learning_map'])
    return points, pred_result

def metric(root_path, save_root_path):
    val_result_path = os.path.join(save_root_path, '08/predictions/')
    label_path = os.path.join(root_path, '08/labels/')

    lable_files = os.listdir(label_path)
    lable_files = sorted(lable_files)

    with open('datasets/semantic-kitti.yaml', 'r') as f:
        task_cfg = yaml.load(f)
    criterion_cate = MultiClassMetric(['static', 'moving'])

    for id, file_name in tqdm.tqdm(enumerate(lable_files)):
        pcds_label = np.fromfile(label_path + file_name.split('.')[0] + '.label', dtype=np.uint32)
        label = utils.relabel(pcds_label & 0xFFFF, task_cfg['learning_map']).astype('int32')
        label = torch.Tensor(label).cuda()

        val_pred = np.fromfile(val_result_path + file_name.split('.')[0] + '.label', dtype=np.uint32)
        pred = utils.relabel(val_pred & 0xFFFF, task_cfg['learning_map']).astype('int32')
        pred = torch.Tensor(pred).cuda().to(torch.int64)
        pred = torch.nn.functional.one_hot(pred, num_classes=-1)
        criterion_cate.addBatch(label, pred)

    metric_cate = criterion_cate.get_metric()
    string = 'Best Epoch'
    for key in metric_cate:
        string = string + '; ' + key + ': ' + str(metric_cate[key])
    print(string + '\n')

parser = argparse.ArgumentParser(description='lidar segmentation')
parser.add_argument('--config', type=str)
parser.add_argument('--tag', type=str, default='base')
parser.add_argument('--modal', type=str, default='val')
args = parser.parse_args()
config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
pGen, pDataset, pModel, pOpt = config.get_config()
prefix = pGen.name  # config
save_path = os.path.join("experiments", prefix, args.tag)
#**************************************************************************************************
fov_xyz = [[-50, -50, -4],[50, 50, 2]]
crop_to_fov = tr.Crop(dims=(0, 1, 2), fov=fov_xyz)
frames_num_max = 8
localmap_points_num_perframe_list = []

modal = args.modal
val_list = ['08']
test_list = ['11', '12', '13', '14', '15', '16',
             '17', '18', '19', '20', '21']

with open('datasets/semantic-kitti.yaml', 'r') as f:
    task_cfg = yaml.load(f)

root_path = 'SemanticKITTI/dataset/sequences'
if modal == 'val':
    pred_root_path = os.path.join(save_path, 'val_results/sequences')
    save_root_path = os.path.join(save_path, 'refine_val_results/sequences')
    sequence_list = val_list
else:
    pred_root_path = os.path.join(save_path, 'test_results/sequences')
    save_root_path = os.path.join(save_path, 'refine_test_results/sequences')
    sequence_list = test_list

time_list = []
for test_sequence in sequence_list:
    print('Sequence id is', test_sequence)
    data_path = os.path.join(root_path, test_sequence, 'velodyne/')
    calib_path = os.path.join(root_path, test_sequence, 'calib.txt')
    pose_path = os.path.join(root_path, test_sequence, 'poses.txt')
    pred_path = os.path.join(pred_root_path, test_sequence, 'predictions/')
    save_path = os.path.join(save_root_path, test_sequence, 'predictions/')

    files = os.listdir(data_path)
    files = sorted(files)

    calib = utils.parse_calibration(calib_path)
    poses_list = utils.parse_poses(pose_path, calib)

    for id, file_name in tqdm.tqdm(enumerate(files)):
        if id >= frames_num_max:
            current_points, current_pred_result = get_data(data_path, pred_path, file_name)
            current_pose_inv = np.linalg.inv(poses_list[id])

            history_points_list, history_pred_result_list = [], []
            for history_id in np.arange(id - 1, id - frames_num_max - 1, -1):
                history_points, history_pred_result = get_data(data_path, pred_path, files[history_id])
                history_pose = poses_list[history_id]

                pose_diff = current_pose_inv.dot(history_pose)
                history_points = utils.Trans(history_points, pose_diff)

                history_points_list.append(history_points)
                history_pred_result_list.append(history_pred_result)

            history_points = np.concatenate((history_points_list), axis=0)
            history_pred_result = np.concatenate((history_pred_result_list), axis=0)
        else:
            current_points, current_pred_result = get_data(data_path, pred_path, file_name)
            current_pose_inv = np.linalg.inv(poses_list[id])

            history_id_list = np.arange(0, frames_num_max, 1)
            history_id_list = np.delete(history_id_list, id)

            history_points_list, history_pred_result_list = [], []
            for history_id in history_id_list:
                history_points, history_pred_result = get_data(data_path, pred_path, files[history_id])
                history_pose = poses_list[history_id]

                pose_diff = current_pose_inv.dot(history_pose)
                history_points = utils.Trans(history_points, pose_diff)

                history_points_list.append(history_points)
                history_pred_result_list.append(history_pred_result)

            history_points = np.concatenate((history_points_list), axis=0)
            history_pred_result = np.concatenate((history_pred_result_list), axis=0)

        current_points_orin = copy.deepcopy(current_points)
        current_pred_result_orin = copy.deepcopy(current_pred_result)

        history_points = torch.tensor(history_points).cuda()
        history_pred_result = torch.tensor(history_pred_result.astype('uint8')).cuda()

        current_points = torch.tensor(current_points).cuda()
        current_pred_result = torch.tensor(current_pred_result.astype('uint8')).cuda()

        #------ crop ------#
        history_points, history_pred_result, _ = crop_to_fov(history_points, history_pred_result)
        current_points, current_pred_result, mask = crop_to_fov(current_points, current_pred_result)
        history_points_num = len(history_points)  # pre store
        #------ concat ------#
        local_map_points = torch.cat((history_points, current_points), dim=0)
        local_map_prediction = torch.cat((history_pred_result, current_pred_result), dim=0)

        #########################
        size = (512, 512, 30)
        pcds_coord_voxel = Quantize(local_map_points,
                                    range_x=(-50.0, 50.0),
                                    range_y=(-50.0, 50.0),
                                    range_z=(-4.0, 2.0),
                                    size=size)
        pcds_coord_cur = pcds_coord_voxel[history_points_num:]
        voxel_label = determine_voxel_labels(pcds_coord_voxel.to(torch.int64),
                                             local_map_prediction.to(torch.int64),
                                             size)
        pred_result_new = get_point_labels_from_voxel_labels(pcds_coord_cur.to(torch.int64), voxel_label, size)
        current_pred_result_orin[mask.cpu().numpy()] = pred_result_new.cpu().numpy()
        #------ save ------#
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pred_map = map(current_pred_result_orin, task_cfg['learning_map_inv'])
        pred_map.tofile(os.path.join(save_path + file_name.split('.')[0] + '.label'))
# ------ metric ------#
if modal == 'val':
    metric(root_path, save_root_path)