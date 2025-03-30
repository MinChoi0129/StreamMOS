import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbone
import pdb
from networks.transformer import TransformerDecoderLayer, TransformerDecoder, PositionEmbeddingLearned
from deformattn.modules import MSDeformAttn
import copy
import deep_point

def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    voxel_feat = deep_point.VoxelMaxPool(pcds_feat=pcds_feat.float(), pcds_ind=pcds_ind, output_size=output_size, scale_rate=scale_rate).to(pcds_feat.dtype)
    return voxel_feat

class Merge(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(Merge, self).__init__()
        cin = cin_low + cin_high
        self.merge_layer = nn.Sequential(
                    backbone.conv3x3(cin, cin // 2, stride=1, dilation=1),
                    nn.BatchNorm2d(cin // 2),
                    backbone.act_layer,
                    
                    backbone.conv3x3(cin // 2, cout, stride=1, dilation=1),
                    nn.BatchNorm2d(cout),
                    backbone.act_layer
                )
        self.scale_factor = scale_factor
    
    def forward(self, x_low, x_high):
        x_high_up = F.upsample(x_high, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        x_merge = torch.cat((x_low, x_high_up), dim=1)
        x_merge = F.dropout(x_merge, p=0.2, training=self.training, inplace=False)
        x_out = self.merge_layer(x_merge)
        return x_out


class AttMerge(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(AttMerge, self).__init__()
        self.scale_factor = scale_factor
        self.cout = cout

        self.att_layer = nn.Sequential(
            backbone.conv3x3(2 * cout, cout // 2, stride=1, dilation=1),
            nn.BatchNorm2d(cout // 2),
            backbone.act_layer,
            backbone.conv3x3(cout // 2, 2, stride=1, dilation=1, bias=True)
        )

        self.conv_high = nn.Sequential(
            backbone.conv3x3(cin_high, cout, stride=1, dilation=1),
            nn.BatchNorm2d(cout),
            backbone.act_layer
        )

        self.conv_low = nn.Sequential(
            backbone.conv3x3(cin_low, cout, stride=1, dilation=1),
            nn.BatchNorm2d(cout),
            backbone.act_layer
        )
    
    def forward(self, x_low, x_high):
        #pdb.set_trace()
        batch_size = x_low.shape[0]
        H = x_low.shape[2]
        W = x_low.shape[3]

        x_high_up = F.upsample(x_high, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        x_merge = torch.stack((self.conv_low(x_low), self.conv_high(x_high_up)), dim=1) #(BS, 2, channels, H, W)
        x_merge = F.dropout(x_merge, p=0.2, training=self.training, inplace=False)

        # attention fusion
        ca_map = self.att_layer(x_merge.view(batch_size, 2*self.cout, H, W))
        ca_map = ca_map.view(batch_size, 2, 1, H, W)
        ca_map = F.softmax(ca_map, dim=1)

        x_out = (x_merge * ca_map).sum(dim=1) #(BS, channels, H, W)
        return x_out


class BEVNet(nn.Module):
    def __init__(self, base_block, context_layers, layers, use_att):
        super(BEVNet, self).__init__()
        #encoder
        self.header = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[0], context_layers[1], layers[0], stride=2, dilation=1, use_att=use_att)
        self.res1 = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[1], context_layers[2], layers[1], stride=2, dilation=1, use_att=use_att)
        self.res2 = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[2], context_layers[3], layers[2], stride=2, dilation=1, use_att=use_att)

        #decoder
        fusion_channels2 = context_layers[3] + context_layers[2]
        self.up2 = AttMerge(context_layers[2], context_layers[3], fusion_channels2 // 2, scale_factor=2)
        
        fusion_channels1 = fusion_channels2 // 2 + context_layers[1]
        self.up1 = AttMerge(context_layers[1], fusion_channels2 // 2, fusion_channels1 // 2, scale_factor=2)
        
        self.out_channels = fusion_channels1 // 2
    
    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=1, dilation=1, use_att=True):
        layer = []
        layer.append(backbone.DownSample2D(in_planes, out_planes, stride=stride))  # DownSample
        
        for i in range(num_blocks):
            layer.append(block(out_planes, dilation=dilation, use_att=False))
        
        layer.append(block(out_planes, dilation=dilation, use_att=True))
        return nn.Sequential(*layer)
    
    def forward(self, x):
        #pdb.set_trace()
        #encoder
        x0 = self.header(x)  # DownSample2D, BasicBlock, BasicBlock, BasicBlock, (BasicBlock+ChannelAttn) 1/2
        x1 = self.res1(x0)  # 1/4
        x2 = self.res2(x1)  # 1/8
        
        #decoder
        x_merge1 = self.up2(x1, x2)  # 1/4
        x_merge = self.up1(x0, x_merge1)  # 1/2
        return x_merge


class CENet(nn.Module):
    def __init__(self, base_block, context_layers, layers, use_att):
        super(CENet, self).__init__()
        # encoder
        self.header = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[0], context_layers[1],
                                       layers[0], stride=2, dilation=1, use_att=use_att)
        self.res1 = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[1], context_layers[2],
                                     layers[1], stride=2, dilation=1, use_att=use_att)
        self.res2 = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[2], context_layers[3],
                                     layers[2], stride=2, dilation=1, use_att=use_att)

        # decoder
        fusion_channels2 = context_layers[3] + context_layers[2]
        self.up2 = AttMerge(context_layers[2], context_layers[3], fusion_channels2 // 2, scale_factor=2)

        fusion_channels1 = fusion_channels2 // 2 + context_layers[1]
        self.up1 = AttMerge(context_layers[1], fusion_channels2 // 2, fusion_channels1 // 2, scale_factor=2)

        self.out_channels = fusion_channels1 // 2

        self.aux = True
        # self.conv_1 = BasicConv2d(416, 256, kernel_size=3, padding=1)
        self.conv_1 = BasicConv2d(224, 128, kernel_size=3, padding=1)
        # self.conv_2 = BasicConv2d(256, self.out_channels, kernel_size=3, padding=1)
        self.conv_2 = BasicConv2d(128, self.out_channels, kernel_size=3, padding=1)
        if self.aux:
            nclasses = 3
            self.aux_head1 = nn.Conv2d(32, nclasses, 1)
            self.aux_head2 = nn.Conv2d(64, nclasses, 1)
            self.aux_head3 = nn.Conv2d(128, nclasses, 1)

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=1, dilation=1, use_att=True):
        layer = []
        layer.append(backbone.DownSample2D(in_planes, out_planes, stride=stride))  # DownSample

        for i in range(num_blocks):
            layer.append(block(out_planes, dilation=dilation, use_att=False))

        layer.append(block(out_planes, dilation=dilation, use_att=True))
        return nn.Sequential(*layer)

    def forward(self, x, return_last_layer=False):
        # pdb.set_trace()
        # encoder
        x0 = self.header(x)  # DownSample2D, BasicBlock, BasicBlock, BasicBlock, (BasicBlock+ChannelAttn) 1/2
        x1 = self.res1(x0)  # 1/4
        x2 = self.res2(x1)  # 1/8

        # decoder
        # res_0 = F.interpolate(x0, size=x.size()[2:], mode='bilinear', align_corners=True)
        # res_1 = F.interpolate(x1, size=x.size()[2:], mode='bilinear', align_corners=True)
        # res_2 = F.interpolate(x2, size=x.size()[2:], mode='bilinear', align_corners=True)
        # res = [x, res_0, res_1, res_2]

        res_0 = F.interpolate(x0, size=x0.size()[2:], mode='bilinear', align_corners=True)
        res_1 = F.interpolate(x1, size=x0.size()[2:], mode='bilinear', align_corners=True)
        res_2 = F.interpolate(x2, size=x0.size()[2:], mode='bilinear', align_corners=True)
        res = [res_0, res_1, res_2]

        out = torch.cat(res, dim=1)
        out = self.conv_1(out)
        out = self.conv_2(out)

        if self.aux:
            res_0 = self.aux_head1(res_0)
            res_1 = self.aux_head2(res_1)
            res_2 = self.aux_head3(res_2)

        if return_last_layer:
            return out, res_0, res_1, res_2, x2
        else:
            return out, res_0, res_1, res_2


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(d_model,  # d_model -> hidden_dim
                                                nhead,  # head number
                                                dim_feedforward,
                                                dropout,  # dropout
                                                activation,  # relu
                                                normalize_before)  # False
        decoder_norm = nn.LayerNorm(d_model)  # LayerNorm
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [channel, bs, c]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # [channel, bs, c]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [query_num, bs, c]
        mask = mask.flatten(1)
        tgt = torch.zeros_like(query_embed)

        # ======== decoder ========
        hs = self.decoder(tgt,
                          src,
                          memory_key_padding_mask=mask,
                          pos=pos_embed,
                          query_pos=query_embed)
        hs = hs[0].permute(1, 2, 0)
        hs = hs.reshape(bs, c, h, w)
        return hs

class DeformAttnModule(nn.Module):
    def __init__(self, deformattn_layers, num_layers):
        super().__init__()
        self.deformattn_layers = self._get_clones(deformattn_layers, num_layers)
        self.num_layers = num_layers

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, query, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.deformattn_layers):
            query = layer(query, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        output = query
        return output

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class DeformAttnLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropoutx = nn.Dropout(dropout)
        self.normx = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, query, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.cross_attn(self.with_pos_embed(query, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        query = query + self.dropout1(src2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query

class CENet_Transformer(nn.Module):
    def __init__(self, base_block, context_layers, layers, nclasses, use_att):
        super(CENet_Transformer, self).__init__()
        query_size = 64
        hidden_dim = 128
        D_MODEL = 128
        DIM_FEEDFORWARD = 512
        DROPOUT = 0.0
        N_LEVELS = 1
        N_HEADS = 4
        N_POINTS = 4
        NUM_ENCODER_LAYERS = 2
        deformattn_layer = DeformAttnLayer(d_model = D_MODEL,
                                           d_ffn = DIM_FEEDFORWARD,
                                           dropout = DROPOUT,
                                           n_levels= N_LEVELS,
                                           n_heads = N_HEADS,
                                           n_points = N_POINTS)  # d_model -> hidden_dim
        self.deformattn_module = DeformAttnModule(deformattn_layer, NUM_ENCODER_LAYERS)
        self.query_embed = nn.Embedding(query_size * query_size, hidden_dim)

        self.header_unbalance_conv = Unbalance_BasicBlock(inplanes=32, kernel_size=(7, 3), padding=(3, 1))
        self.res1_unbalance_conv = Unbalance_BasicBlock(inplanes=64, kernel_size=(5, 3), padding=(2, 1))
        # encoder
        self.header_bev = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[0], context_layers[1],
                                           layers[0], stride=2, dilation=1, use_att=use_att)
        self.header_bev[1] = self.header_unbalance_conv
        self.header_rv = self._make_layer(eval('backbone.{}'.format(base_block)), 32, context_layers[1],
                                          layers[0]-1, stride=1, dilation=1, use_att=use_att)
        self.res1_bev = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[1] * 2, context_layers[2],
                                     layers[1], stride=2, dilation=1, use_att=use_att)
        self.res1_bev[1] = self.res1_unbalance_conv
        self.res1_rv = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[1] * 2, context_layers[2],
                                     layers[1]-1, stride=1, dilation=1, use_att=use_att)
        self.res2 = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[2] * 2, context_layers[3],
                                     layers[2], stride=2, dilation=1, use_att=use_att)

        # decoder
        fusion_channels2 = context_layers[3] + context_layers[2]
        self.up2 = AttMerge(context_layers[2], context_layers[3], fusion_channels2 // 2, scale_factor=2)

        fusion_channels1 = fusion_channels2 // 2 + context_layers[1]
        self.up1 = AttMerge(context_layers[1], fusion_channels2 // 2, fusion_channels1 // 2, scale_factor=2)

        self.out_channels = fusion_channels1 // 2

        self.aux = True
        self.conv_1 = BasicConv2d(320, 128, kernel_size=3, padding=1)
        self.conv_2 = BasicConv2d(128, self.out_channels, kernel_size=3, padding=1)
        if self.aux:
            self.aux_head1 = nn.Conv2d(64, nclasses, 1)
            self.aux_head2 = nn.Conv2d(128, nclasses, 1)
            self.aux_head3 = nn.Conv2d(128, nclasses, 1)

        self.bev_grid2point_x0 = backbone.BilinearSample(in_dim=4, scale_rate=(0.5, 0.5))
        self.bev_grid2point_x1 = backbone.BilinearSample(in_dim=4, scale_rate=(0.25, 0.25))

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=1, dilation=1, use_att=True):
        layer = []
        layer.append(backbone.DownSample2D(in_planes, out_planes, stride=stride))  # DownSample

        for i in range(num_blocks):
            layer.append(block(out_planes, dilation=dilation, use_att=False))

        layer.append(block(out_planes, dilation=dilation, use_att=True))
        return nn.Sequential(*layer)

    def forward(self, x, pcds_cood_cur, pcds_sphere_coord_cur, query_embed_store, use_query_store=False, return_query=False):
        # pdb.set_trace()
        # encoder
        x0 = self.header_bev(x)  # DownSample2D, BasicBlock, BasicBlock, BasicBlock, (BasicBlock+ChannelAttn) 1/2

        x0_point = self.bev_grid2point_x0(x0, pcds_cood_cur)
        x0_rv = VoxelMaxPool(pcds_feat=x0_point,
                             pcds_ind=pcds_sphere_coord_cur,
                             output_size=(32, 1024), scale_rate=(0.5, 0.5))  # (BS*T, C, H, W)
        x0_rv = self.header_rv(x0_rv)
        x0_point = self.bev_grid2point_x0(x0_rv, pcds_sphere_coord_cur)

        x0_bev = VoxelMaxPool(pcds_feat=x0_point,
                             pcds_ind=pcds_cood_cur,
                             output_size=(256, 256), scale_rate=(0.5, 0.5))  # (BS*T, C, H, W)
        x0 = torch.cat((x0, x0_bev), dim=1)

        #############################
        x1 = self.res1_bev(x0)  # 1/4

        x1_point = self.bev_grid2point_x1(x1, pcds_cood_cur)
        x1_rv = VoxelMaxPool(pcds_feat=x1_point,
                             pcds_ind=pcds_sphere_coord_cur,
                             output_size=(16, 512), scale_rate=(0.25, 0.25))  # (BS*T, C, H, W)
        x1_rv = self.res1_rv(x1_rv)
        x1_point = self.bev_grid2point_x1(x1_rv, pcds_sphere_coord_cur)

        x1_bev = VoxelMaxPool(pcds_feat=x1_point,
                             pcds_ind=pcds_cood_cur,
                             output_size=(128, 128), scale_rate=(0.25, 0.25))  # (BS*T, C, H, W)
        x1 = torch.cat((x1, x1_bev), dim=1)

        #############################
        x2 = self.res2(x1)  # 1/8

        # Deform
        bs, c, x, y = x2.shape
        spatial_shapes = (x, y)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=x2.device).unsqueeze(0)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.ones((bs, 1, 2),dtype=x2.dtype, device=x2.device)

        src_flatten = x2.flatten(2).transpose(2, 1)
        if use_query_store:
            query_embed = query_embed_store.flatten(2).transpose(2, 1)
        else:
            query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        src_flatten = self.deformattn_module(query_embed, src_flatten, spatial_shapes, level_start_index, valid_ratios)
        x2 = src_flatten.transpose(2, 1).reshape(bs, c, x, y)

        res_0 = F.interpolate(x0, size=x0.size()[2:], mode='bilinear', align_corners=True)
        res_1 = F.interpolate(x1, size=x0.size()[2:], mode='bilinear', align_corners=True)
        res_2 = F.interpolate(x2, size=x0.size()[2:], mode='bilinear', align_corners=True)
        res = [res_0, res_1, res_2]

        out = torch.cat(res, dim=1)
        out = self.conv_1(out)
        out = self.conv_2(out)

        if self.aux:
            res_0 = self.aux_head1(res_0)
            res_1 = self.aux_head2(res_1)
            res_2 = self.aux_head3(res_2)

        if return_query:
            return out, x1_point, res_0, res_1, res_2, x2
        else:
            return out, res_0, res_1, res_2

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super(BasicConv2d, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

class Unbalance_BasicBlock(nn.Module):
    def __init__(self, inplanes, kernel_size, padding):
        super(Unbalance_BasicBlock, self).__init__()
        self.layer7x3 = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel_size=(kernel_size[0], kernel_size[1]), stride=(1, 1), padding=(padding[0], padding[1]), dilation=1, bias=False),
                                      nn.BatchNorm2d(inplanes),
                                      nn.ReLU(inplace=True))
        self.layer3x7 = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel_size=(kernel_size[1], kernel_size[0]), stride=(1, 1), padding=(padding[1], padding[0]), dilation=1, bias=False),
                                      nn.BatchNorm2d(inplanes),
                                      nn.ReLU(inplace=True))
        self.layer3x3 = nn.Sequential(nn.Conv2d(inplanes * 2, inplanes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=1, bias=False),
                                      nn.BatchNorm2d(inplanes))

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out_7x3 = self.layer7x3(x)
        out_3x7 = self.layer3x7(x)
        out = self.layer3x3(torch.cat((out_7x3, out_3x7), dim=1))
        out = self.act(out + x)
        return out