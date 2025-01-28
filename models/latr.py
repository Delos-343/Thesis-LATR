import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from mmdet3d.models import build_backbone, build_neck
from .latr_head import LATRHead
from mmcv.utils import Config
from .ms2one import build_ms2one
from .utils import deepFeatureExtractor_EfficientNet
from mmdet.models.builder import BACKBONES


class LATR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.no_cuda = args.no_cuda
        self.batch_size = args.batch_size
        self.num_lane_type = 1  # No centerline
        self.num_y_steps = args.num_y_steps
        self.max_lanes = args.max_lanes
        self.num_category = args.num_category

        # Ensure latr_cfg exists before accessing attributes
        if not hasattr(args, 'latr_cfg'):
            raise ValueError("LATR configuration (latr_cfg) missing in args.")

        _dim_ = args.latr_cfg.get('fpn_dim', 256)  # Default to 256 if not provided
        num_query = args.latr_cfg.get('num_query', 100)
        num_group = args.latr_cfg.get('num_group', 4)
        sparse_num_group = args.latr_cfg.get('sparse_num_group', 1)

        # Backbone initialization
        if hasattr(args.latr_cfg, 'encoder'):
            self.encoder = build_backbone(args.latr_cfg.encoder)
            self.encoder.init_weights()
        else:
            raise ValueError("Encoder configuration missing in latr_cfg.")

        # Neck initialization (optional)
        if hasattr(args.latr_cfg, 'neck') and args.latr_cfg.neck:
            self.neck = build_neck(args.latr_cfg.neck)
        else:
            self.neck = None

        # Multi-scale feature transformation
        if hasattr(args, 'ms2one'):
            self.ms2one = build_ms2one(args.ms2one)
        else:
            raise ValueError("ms2one configuration missing in args.")

        # LATR Head initialization with safety checks
        if hasattr(args, 'transformer') and hasattr(args, 'sparse_ins_decoder'):
            self.head = LATRHead(
                args=args,
                dim=_dim_,
                num_group=num_group,
                num_convs=4,
                in_channels=_dim_,
                kernel_dim=_dim_,
                position_range=getattr(args, 'position_range', None),
                top_view_region=getattr(args, 'top_view_region', None),
                positional_encoding=dict(
                    type='SinePositionalEncoding',
                    num_feats=_dim_ // 2, normalize=True),
                num_query=num_query,
                pred_dim=self.num_y_steps,
                num_classes=self.num_category,
                embed_dims=_dim_,
                transformer=args.transformer,
                sparse_ins_decoder=args.sparse_ins_decoder,
                **args.latr_cfg.get('head', {}),
                trans_params=args.latr_cfg.get('trans_params', {})
            )
        else:
            raise ValueError("Transformer or sparse_ins_decoder configuration missing in args.")

    def forward(self, image, _M_inv=None, is_training=True, extra_dict=None):

        if extra_dict is None:
            raise ValueError("extra_dict must be provided in forward pass.")
        
        extra_dict['image'] = image  # Add image to dictionary

        # Ensure encoder output is valid
        out_featList = self.encoder(image)
        
        if self.neck:
            neck_out = self.neck(out_featList)
        else:
            neck_out = out_featList

        neck_out = self.ms2one(neck_out)

        # Ensure required keys exist in extra_dict
        required_keys = ['seg_idx_label', 'seg_label', 'lidar2img', 'pad_shape']
        missing_keys = [key for key in required_keys if key not in extra_dict]

        if missing_keys:
            raise ValueError(f"Missing keys in extra_dict: {missing_keys}")

        # Set default values for optional training-related keys
        ground_lanes = extra_dict.get('ground_lanes', None) if is_training else None
        ground_lanes_dense = extra_dict.get('ground_lanes_dense', None) if is_training else None

        # Ensure image is explicitly passed
        extra_dict['image'] = image

        output = self.head(
            dict(
                x=neck_out,
                lane_idx=extra_dict['seg_idx_label'],
                seg=extra_dict['seg_label'],
                lidar2img=extra_dict['lidar2img'],
                pad_shape=extra_dict['pad_shape'],
                ground_lanes=ground_lanes,
                ground_lanes_dense=ground_lanes_dense,
                image=image,  # Explicitly passing image
            ),
            is_training=is_training,
        )
        return output
