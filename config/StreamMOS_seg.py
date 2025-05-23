def get_config():
    class General:
        log_frequency = 100
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_size_per_gpu = 4
        fp16 = False

        SeqDir = 'SemanticKITTI/dataset/sequences'
        category_list = ['static', 'moving']

        loss_mode = 'ohem'
        K = 2
        class Voxel:
            RV_theta = (-25.0, 3.0)
            range_x = (-50.0, 50.0)
            range_y = (-50.0, 50.0)
            range_z = (-4.0, 2.0)

            bev_shape = (512, 512, 30)
            rv_shape = (64, 2048)

    class DatasetParam:
        class Train:
            data_src = 'data_StreamMOS_seg'
            drop_few_static_frames = False
            num_workers = 4
            frame_point_num = 130000
            SeqDir = General.SeqDir
            Voxel = General.Voxel
            seq_num = General.K + 1
            class CopyPasteAug:
                is_use = True
                ObjBackDir = 'object_bank_semkitti'
                paste_max_obj_num = 20
            class AugParam:
                noise_mean = 0
                noise_std = 0.0001
                theta_range = (-180.0, 180.0)
                shift_range = ((-3, 3), (-3, 3), (-0.4, 0.4))
                size_range = (0.95, 1.05)

        class Val:
            data_src = 'data_StreamMOS_seg'
            drop_few_static_frames = True
            num_workers = 4
            frame_point_num = 160000
            SeqDir = General.SeqDir
            Voxel = General.Voxel
            seq_num = General.K + 1

        class Test:
            data_src = 'data_test_StreamMOS'
            num_workers = 4
            frame_point_num = 160000
            SeqDir = General.SeqDir
            Voxel = General.Voxel
            seq_num = General.K + 1
            learning_map_inv = {0: 0, 1: 9, 2: 251}

    class ModelParam:
        prefix = "StreamMOS_seg.AttNet"
        Voxel = General.Voxel
        category_list = General.category_list
        class_num = len(category_list) + 1
        loss_mode = General.loss_mode
        seq_num = General.K + 1

        point_feat_out_channels = 64
        fusion_mode = 'CatFusion'

        class BEVParam:
            base_block = 'BasicBlock'
            context_layers = [64, 32, 64, 128]
            layers = [2, 3, 4]
            bev_grid2point = dict(type='BilinearSample', scale_rate=(0.5, 0.5))

        class pretrain:
            pretrain_epoch = 40

    class OptimizeParam:
        class optimizer:
            type = "sgd"
            base_lr = 0.02
            momentum = 0.9
            nesterov = True
            wd = 1e-3

        class schedule:
            type = "step"
            begin_epoch = 0
            end_epoch = 10
            pct_start = 0.01
            final_lr = 1e-6
            step = 2
            decay_factor = 0.1

    return General, DatasetParam, ModelParam, OptimizeParam