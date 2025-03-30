import copy
import numpy as np
import torch
import utils.transforms as tr
import os
import yaml
import datasets.utils as utils
import scipy
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay, ConvexHull
from multiprocessing import Pool
from tqdm import tqdm
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

def min_bounding_box_3d(points):
    points = np.array(points)
    hull = ConvexHull(points)
    indices = hull.vertices
    convex_hull_points = points[indices]
    min_x, min_y, min_z = np.min(convex_hull_points, axis=0)
    max_x, max_y, max_z = np.max(convex_hull_points, axis=0)
    corners = np.array([
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, max_z],
        [min_x, max_y, max_z]
    ])

    return corners

def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag

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
    x = pcds[:, 0].copy()
    y = pcds[:, 1].copy()
    z = pcds[:, 2].copy()

    size_x = size[0]
    size_y = size[1]
    size_z = size[2]

    dx = (range_x[1] - range_x[0]) / size_x
    dy = (range_y[1] - range_y[0]) / size_y
    dz = (range_z[1] - range_z[0]) / size_z

    x_quan = ((x - range_x[0]) / dx)
    y_quan = ((y - range_y[0]) / dy)
    z_quan = ((z - range_z[0]) / dz)

    pcds_quan = np.stack((x_quan, y_quan, z_quan), axis=-1)
    return pcds_quan

def get_data(data_path, pred_path, file_name):
    points = np.fromfile(data_path + file_name, dtype=np.float32).reshape((-1, 4))
    pcds_label = np.fromfile(pred_path + file_name.split('.')[0] + '.label', dtype=np.uint32)
    sem_label = pcds_label & 0xFFFF
    pred_result = utils.relabel(sem_label, task_cfg['learning_map'])
    return points, pred_result

def cluster(current_points_orin, current_pred_result_orin, current_pred_bf_result_orin, local_map_points, local_map_prediction):
    foreground_index = np.where(current_pred_bf_result_orin == 2)[0]
    if len(foreground_index) == 0:
        return current_pred_result_orin
    foreground_points = current_points_orin[foreground_index][:, :3]

    dbscan_radius = 0.3
    dbscan = DBSCAN(eps=dbscan_radius, min_samples=5)

    points_labels = dbscan.fit_predict(foreground_points)
    labels = np.unique(points_labels)
    cluster_points_list = []
    cluster_points_index_list = []
    cluster_centers_list = []
    for label in labels:
        if label != -1:
            cluster_points = foreground_points[points_labels == label]
            cluster_points_index = foreground_index[points_labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            if len(cluster_points) > 30:
                cluster_points_list.append(cluster_points)
                cluster_centers_list.append(cluster_center)
                cluster_points_index_list.append(cluster_points_index)

    cluster_label_list = []
    for idx in range(len(cluster_centers_list)):
        cluster_points = cluster_points_list[idx]
        cluster_corners = min_bounding_box_3d(cluster_points)
        ########
        z_min = np.min(cluster_corners[:, -1])
        z_min_flag = np.where(cluster_corners[:, -1] == z_min)
        cluster_corners[z_min_flag, -1] += 0.2
        ########
        flag = in_hull(local_map_points[:, :3], cluster_corners)

        cluster_local_points = local_map_points[flag]
        cluster_local_points_prediction = local_map_prediction[flag]

        static_points_num = sum(cluster_local_points_prediction[cluster_local_points_prediction == 1])
        dynamic_points_num = sum(cluster_local_points_prediction[cluster_local_points_prediction == 2])
        if dynamic_points_num > static_points_num:
            cluster_label_list.append(2)
        else:
            cluster_label_list.append(1)

    for id in range(len(cluster_label_list)):
        cluster_label = cluster_label_list[id]
        current_pred_result_orin[cluster_points_index_list[id]] = cluster_label

    return current_pred_result_orin

def post_processing(id):
    if id >= frames_num_max:
        current_points, current_pred_result = get_data(data_path, pred_path, files[id])  # data
        current_pred_bf_result = np.fromfile(pred_bf_path + files[id].split('.')[0] + '.label', dtype=np.uint32)
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
        current_points, current_pred_result = get_data(data_path, pred_path, files[id])  # data
        current_pred_bf_result = np.fromfile(pred_bf_path + files[id].split('.')[0] + '.label', dtype=np.uint32)
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

    # ------ crop ------#
    history_points, history_pred_result, _ = crop_to_fov(history_points, history_pred_result)
    current_points_orin = copy.deepcopy(current_points)  # copy
    current_pred_result_orin = copy.deepcopy(current_pred_result)
    current_points, current_pred_result, mask = crop_to_fov(current_points, current_pred_result)
    history_points_num = len(history_points)  # pre store

    # ------ concat ------#
    local_map_points = np.concatenate((history_points, current_points), axis=0)
    local_map_prediction = np.concatenate((history_pred_result, current_pred_result), axis=0).astype('int64')

    #########################
    size = (512, 512, 30)
    pcds_coord_voxel = Quantize(local_map_points,
                                range_x=(-50.0, 50.0),
                                range_y=(-50.0, 50.0),
                                range_z=(-4.0, 2.0),
                                size=size)
    pcds_coord_voxel = torch.tensor(pcds_coord_voxel).cuda().to(torch.int64)
    pcds_coord_cur = pcds_coord_voxel[history_points_num:]
    voxel_label = determine_voxel_labels(pcds_coord_voxel,
                                         torch.tensor(local_map_prediction).cuda(),
                                         size)
    pred_result_new = get_point_labels_from_voxel_labels(pcds_coord_cur, voxel_label, size)
    current_pred_result_orin[mask] = pred_result_new.cpu().numpy()
    #########################
    current_pred_result_orin = cluster(current_points_orin, current_pred_result_orin, current_pred_bf_result,
                                       local_map_points, local_map_prediction)
    #########################

    # ------ save ------#
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pred_map = map(current_pred_result_orin, task_cfg['learning_map_inv'])
    pred_map.tofile(os.path.join(save_path + files[id].split('.')[0] + '.label'))

def metric(root_path, save_root_path):
    val_result_path = os.path.join(save_root_path, '08/predictions/')
    label_path = os.path.join(root_path, '08/labels/')

    lable_files = os.listdir(label_path)
    lable_files = sorted(lable_files)

    with open('datasets/semantic-kitti.yaml', 'r') as f:
        task_cfg = yaml.load(f)
    criterion_cate = MultiClassMetric(['static', 'moving'])

    for id, file_name in tqdm(enumerate(lable_files)):
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
    pred_bf_root_path = os.path.join(save_path, 'val_bf_results/sequences')
    save_root_path = os.path.join(save_path, 'refine_val_results/sequences')
    sequence_list = val_list
else:
    pred_root_path = os.path.join(save_path, 'test_results/sequences')
    pred_bf_root_path = os.path.join(save_path, 'test_bf_results/sequences')
    save_root_path = os.path.join(save_path, 'refine_test_results/sequences')
    sequence_list = test_list

for test_sequence in sequence_list:
    print('Sequence id is', test_sequence)
    data_path = os.path.join(root_path, test_sequence, 'velodyne/')
    calib_path = os.path.join(root_path, test_sequence, 'calib.txt')
    pose_path = os.path.join(root_path, test_sequence, 'poses.txt')
    pred_path = os.path.join(pred_root_path, test_sequence, 'predictions/')
    pred_bf_path = os.path.join(pred_bf_root_path, test_sequence, 'predictions/')
    save_path = os.path.join(save_root_path, test_sequence, 'predictions/')

    files = os.listdir(data_path)
    files = sorted(files)

    calib = utils.parse_calibration(calib_path)
    poses_list = utils.parse_poses(pose_path, calib)

    ###################
    with Pool(8) as p:
        list(tqdm(p.imap(post_processing, range(len(files))), total=len(files)))
# ------ metric ------#
if modal == 'val':
    metric(root_path, save_root_path)