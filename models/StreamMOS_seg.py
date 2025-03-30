import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import backbone, multi_view_encoder
from networks.backbone import get_module
import deep_point

from utils.criterion import CE_OHEM
from utils.lovasz_losses import lovasz_softmax
from utils.boundary_loss import BoundaryLoss
import yaml
import copy
import pdb


def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    voxel_feat = deep_point.VoxelMaxPool(pcds_feat=pcds_feat.float(), pcds_ind=pcds_ind, output_size=output_size, scale_rate=scale_rate).to(pcds_feat.dtype)
    return voxel_feat

class Refine(nn.Module):
    def __init__(self, fusion_mode, point_fusion_channels, point_feat_out_channels, class_num):
        super(Refine, self).__init__()
        self.bf_point_post = eval('backbone.{}'.format(fusion_mode))(in_channel_list=point_fusion_channels, out_channel=point_feat_out_channels)
        self.bf_pred_layer = backbone.PredBranch(point_feat_out_channels, class_num)

    def forward(self, point_feat_tmp_cur, point_bev_feat, point_feat_1):
        bf_point_feat_out = self.bf_point_post(point_feat_tmp_cur, point_bev_feat, point_feat_1)
        bf_pred_cls = self.bf_pred_layer(bf_point_feat_out).float()
        return bf_pred_cls

class AttNet(nn.Module):
    def __init__(self, pModel):
        super(AttNet, self).__init__()
        self.pModel = pModel

        self.bev_shape = list(pModel.Voxel.bev_shape)
        self.rv_shape = list(pModel.Voxel.rv_shape)
        self.bev_wl_shape = self.bev_shape[:2]

        self.dx = (pModel.Voxel.range_x[1] - pModel.Voxel.range_x[0]) / (pModel.Voxel.bev_shape[0])
        self.dy = (pModel.Voxel.range_y[1] - pModel.Voxel.range_y[0]) / (pModel.Voxel.bev_shape[1])
        self.dz = (pModel.Voxel.range_z[1] - pModel.Voxel.range_z[0]) / (pModel.Voxel.bev_shape[2])

        self.point_feat_out_channels = pModel.point_feat_out_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.boundary_loss = BoundaryLoss().to(self.device)

        self.build_network()
        self.build_loss()

    def build_loss(self):
        self.criterion_seg_cate = None
        print("Loss mode: {}".format(self.pModel.loss_mode))
        if self.pModel.loss_mode == 'ce':
            self.criterion_seg_cate = nn.CrossEntropyLoss(ignore_index=0)
        elif self.pModel.loss_mode == 'ohem':
            self.criterion_seg_cate = CE_OHEM(top_ratio=0.2, top_weight=4.0, ignore_index=0)
        elif self.pModel.loss_mode == 'wce':
            content = torch.zeros(self.pModel.class_num, dtype=torch.float32)
            with open('datasets/semantic-kitti.yaml', 'r') as f:
                task_cfg = yaml.load(f)
                for cl, freq in task_cfg["content"].items():
                    x_cl = task_cfg['learning_map'][cl]
                    content[x_cl] += freq

            loss_w = 1 / (content + 0.001)
            loss_w[0] = 0
            print("Loss weights from content: ", loss_w)
            self.criterion_seg_cate = nn.CrossEntropyLoss(weight=loss_w)
        else:
            raise Exception('loss_mode must in ["ce", "wce", "ohem"]')

    def build_network(self):
        # build network
        bev_context_layer = copy.deepcopy(self.pModel.BEVParam.context_layers)
        bev_layers = copy.deepcopy(self.pModel.BEVParam.layers)
        bev_base_block = self.pModel.BEVParam.base_block
        bev_grid2point = self.pModel.BEVParam.bev_grid2point

        fusion_mode = self.pModel.fusion_mode

        point_feat_channels = bev_context_layer[0]
        bev_context_layer[0] = self.pModel.seq_num * bev_context_layer[0]

        # network
        self.point_pre = backbone.PointNetStacker(7, point_feat_channels, pre_bn=True, stack_num=2)
        self.bev_net = multi_view_encoder.CENet_Transformer(bev_base_block, bev_context_layer, bev_layers, self.pModel.class_num, use_att=True)
        self.bev_grid2point = get_module(bev_grid2point, in_dim=self.bev_net.out_channels)

        point_fusion_channels = (point_feat_channels, self.bev_net.out_channels, 64)
        self.point_post = eval('backbone.{}'.format(fusion_mode))(in_channel_list=point_fusion_channels, out_channel=self.point_feat_out_channels)

        self.pred_layer = backbone.PredBranch(self.point_feat_out_channels, self.pModel.class_num)

        self.refine = Refine(fusion_mode, point_fusion_channels, self.point_feat_out_channels, self.pModel.class_num)

    def stage_forward(self, point_feat, pcds_coord, pcds_sphere_coord, query_embed_store=None, use_query_store=False, return_query=False):
        '''
        Input:
            point_feat (BS, T, C, N, 1)
            pcds_coord (BS, T, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord (BS, T, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
        Output:
            point_feat_out (BS, C1, N, 1)
        '''
        BS, T, C, N, _ = point_feat.shape

        pcds_cood_cur = pcds_coord[:, 0, :, :2].contiguous()
        pcds_sphere_coord_cur = pcds_sphere_coord[:, 0].contiguous()

        # BEV
        point_feat_tmp = self.point_pre(point_feat.view(BS*T, C, N, 1))
        bev_input = VoxelMaxPool(pcds_feat=point_feat_tmp, pcds_ind=pcds_coord.view(BS*T, N, 3, 1)[:, :, :2].contiguous(), output_size=self.bev_wl_shape, scale_rate=(1.0, 1.0)) #(BS*T, C, H, W)
        bev_input = bev_input.view(BS, -1, self.bev_wl_shape[0], self.bev_wl_shape[1])
        bev_feat, point_feat_1, pred_res_cls_0, pred_res_cls_1, pred_res_cls_2, query_embed_store = self.bev_net(bev_input, pcds_cood_cur, pcds_sphere_coord_cur, query_embed_store, use_query_store, return_query)
        point_bev_feat = self.bev_grid2point(bev_feat, pcds_cood_cur)

        # merge multi-view
        point_feat_tmp_cur = point_feat_tmp.view(BS, T, -1, N, 1)[:, 0].contiguous()
        point_feat_out = self.point_post(point_feat_tmp_cur, point_bev_feat, point_feat_1)

        # pred
        pred_cls = self.pred_layer(point_feat_out).float()
        bf_pred_cls = self.refine(point_feat_tmp_cur, point_bev_feat, point_feat_1)

        return pred_cls, bf_pred_cls, pred_res_cls_0, pred_res_cls_1, pred_res_cls_2, query_embed_store

    def consistency_loss_l1(self, pred_cls, pred_cls_raw):
        '''
        Input:
            pred_cls, pred_cls_raw (BS, C, N, 1)
        '''
        pred_cls_softmax = F.softmax(pred_cls, dim=1)
        pred_cls_raw_softmax = F.softmax(pred_cls_raw, dim=1)

        loss = (pred_cls_softmax - pred_cls_raw_softmax).abs().sum(dim=1).mean()
        return loss

    def single_forward(self, batch, query_embed_store=None, use_query_store=False, return_query=False):
        '''
        Input:
            pcds_xyzi, pcds_xyzi_raw (BS, T, C, N, 1), C -> (x, y, z, intensity, dist, ...)
            pcds_coord, pcds_coord_raw (BS, T, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord, pcds_sphere_coord_raw (BS, T, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
            pcds_target (BS, N, 1)
        Output:
            loss
        '''
        pcds_target = batch['pcds_target']
        pcds_bf_target = batch['pcds_bf_target']
        pcds_bev_target = batch['pcds_bev_target']
        pred_cls, bf_pred_cls, pred_res_cls_0, pred_res_cls_1, pred_res_cls_2, query_embed_store = self.stage_forward(batch['pcds_xyzi'], batch['pcds_coord'], batch['pcds_sphere_coord'],
                                                                                                                      query_embed_store, use_query_store, return_query)

        bs, time_num, _, _ = pred_cls.shape
        loss = self.criterion_seg_cate(bf_pred_cls, pcds_bf_target) + 3 * lovasz_softmax(bf_pred_cls, pcds_bf_target, ignore=0)

        return loss, query_embed_store

    def forward(self, batch):
        sample_num = 3
        singe_batch = {}
        loss_list = []
        query_embed_store = None
        for i in range(sample_num):
            singe_batch['pcds_target'] = batch['pcds_target' + '_' + str(i)]
            singe_batch['pcds_bf_target'] = batch['pcds_bf_target' + '_' + str(i)]
            singe_batch['pcds_bev_target'] = batch['pcds_bev_target' + '_' + str(i)]
            # singe_batch['pcds_bev_target_raw'] = batch['pcds_bev_target_raw' + '_' + str(i)]
            singe_batch['pcds_xyzi'] = batch['pcds_xyzi' + '_' + str(i)]
            singe_batch['pcds_coord'] = batch['pcds_coord' + '_' + str(i)]
            singe_batch['pcds_sphere_coord'] = batch['pcds_sphere_coord' + '_' + str(i)]
            # singe_batch['pcds_xyzi_raw'] = batch['pcds_xyzi_raw' + '_' + str(i)]
            # singe_batch['pcds_coord_raw'] = batch['pcds_coord_raw' + '_' + str(i)]
            # singe_batch['pcds_sphere_coord_raw'] = batch['pcds_sphere_coord_raw' + '_' + str(i)]

            if i == 0:
                loss, query_embed_store = self.single_forward(singe_batch, return_query=True)
                loss_list.append(loss)
            else:
                loss, query_embed_store = self.single_forward(singe_batch, query_embed_store=query_embed_store, use_query_store=True, return_query=True)
                loss_list.append(loss)

        loss = sum(loss_list) / len(loss_list)
        return loss

    def infer(self, batch, i, query_embed_store=None):
        '''
        Input:
            pcds_xyzi (BS, T, C, N, 1), C -> (x, y, z, intensity, dist, ...)
            pcds_coord (BS, T, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord (BS, T, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
        Output:
            pred_cls, (BS, C, N, 1)
        '''
        if i == 0:
            pred_cls, bf_pred_cls, pred_res_cls_0, pred_res_cls_1, pred_res_cls_2, query_embed_store = self.stage_forward(batch['pcds_xyzi'].squeeze(0),
                                                                                                                          batch['pcds_coord'].squeeze(0),
                                                                                                                          batch['pcds_sphere_coord'].squeeze(0),
                                                                                                                          return_query=True)
        else:
            pred_cls, bf_pred_cls, pred_res_cls_0, pred_res_cls_1, pred_res_cls_2, query_embed_store = self.stage_forward(batch['pcds_xyzi'].squeeze(0),
                                                                                                                          batch['pcds_coord'].squeeze(0),
                                                                                                                          batch['pcds_sphere_coord'].squeeze(0),
                                                                                                                          query_embed_store=query_embed_store,
                                                                                                                          use_query_store=True,
                                                                                                                          return_query=True)
        return pred_cls, bf_pred_cls, pred_res_cls_0, pred_res_cls_1, pred_res_cls_2, query_embed_store