import torch.nn as nn
import torch

from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head import BBoxHead

import torch.nn.functional as F


@HEADS.register_module
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 bnneck=None,
                 exp_ops=None,
                 gap_gmp=None,
                 extra_fc=None,
                 s_fc=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.bnneck = bnneck
        self.exp_ops = exp_ops
        self.gap_gmp = gap_gmp
        self.extra_fc = extra_fc
        self.s_fc = s_fc

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        if self.s_fc is not None:
            self.extra_shared_fcs = nn.ModuleList()
            self.extra_shared_fcs.append(nn.Linear(self.in_channels * self.roi_feat_area, self.fc_out_channels))
            self.extra_shared_fcs.append(nn.Linear(self.fc_out_channels, self.fc_out_channels))

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.extra_fc is not None:
            before_num_classes = 81
            if self.with_cls:
                self.fc_cls = nn.Linear(self.cls_last_dim, before_num_classes)
                self.extra_fc_cls = nn.Linear(before_num_classes, self.num_classes)
            if self.with_reg:
                out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                            self.num_classes)
                self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)
                self.extra_fc_reg = nn.Linear(out_dim_reg, out_dim_reg)
        else:
            if self.with_cls:
                if self.gap_gmp is not None:
                    self.fc_cls = nn.Linear(self.cls_last_dim + self.in_channels, self.num_classes)
                    # self.fc_cls = nn.Linear(self.cls_last_dim + self.in_channels * 2, self.num_classes)
                else:
                    self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
            if self.with_reg:
                out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                            self.num_classes)
                self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

        if self.bnneck is not None:
            self.bnneck_cls = nn.BatchNorm1d(self.cls_last_dim)
            self.bnneck_reg = nn.BatchNorm1d(self.reg_last_dim)
            self.bnneck_cls.bias.requires_grad_(False)
            self.bnneck_reg.bias.requires_grad_(False)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # add by luyi
        if self.gap_gmp is not None:
            x_t = F.adaptive_avg_pool2d(x, 1) + F.adaptive_max_pool2d(x, 1)
            # x_t = torch.cat([F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)], dim=1)
            x_t = x_t.squeeze()

        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            if self.s_fc is not None:
                x_s = x
                for fc in self.extra_shared_fcs:
                    x_s = self.relu(fc(x_s))

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        if self.s_fc is not None:
            x_cls = x_s
        else:
            x_cls = x
        x_reg = x

        # add by luyi
        if self.gap_gmp is not None:
            x_cls = torch.cat([x_cls, x_t], dim=1)

        if self.exp_ops is not None:
            exp_params = self.exp_ops.split('_')
            assert len(exp_params) == 2
            x_cls = getattr(torch, exp_params[0])(x_cls / float(exp_params[1]))
            x_reg = getattr(torch, exp_params[0])(x_reg / float(exp_params[1]))
        if self.bnneck is not None:
            x_cls = self.bnneck_cls(x_cls)
            x_reg = self.bnneck_reg(x_reg)

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        if self.extra_fc is not None:
            cls_score = self.fc_cls(x_cls) if self.with_cls else None
            cls_score = self.extra_fc_cls(cls_score) if self.with_cls else None
            bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
            bbox_pred = self.extra_fc_reg(bbox_pred) if self.with_reg else None
        else:
            cls_score = self.fc_cls(x_cls) if self.with_cls else None
            bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@HEADS.register_module
class SharedFCBBoxHead(ConvFCBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
