import re

import torch

from mmdet.utils import build_from_cfg
from .registry import OPTIMIZERS

# modify by luyi
def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 4 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult,
                  dwconv_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.
                  `dwconv_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of depthwise conv layers.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Example:
        >>> import torch
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> optimizer = build_optimizer(model, optimizer_cfg)
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        params = model.parameters()
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in paramwise_options
                or 'norm_decay_mult' in paramwise_options
                or 'dwconv_decay_mult' in paramwise_options):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        dwconv_decay_mult = paramwise_options.get('dwconv_decay_mult', 1.)

        bbox_head_lr_mult = paramwise_options.get('bbox_head_lr_mult', 1.)
        dcn_lr_mult = paramwise_options.get('dcn_lr_mult', 1.)
        neck_lr_mult = paramwise_options.get('neck_lr_mult', 1.)
        bbox_head_reg_lr_mult = paramwise_options.get('bbox_head_reg_lr_mult', 1.)
        extra_bbox_head_lr_mult = paramwise_options.get('extra_bbox_head_lr_mult', 1.)
        rpn_head_lr_mult = paramwise_options.get('rpn_head_lr_mult', 1.)

        bbox_head_mult_lr_list = list()
        dcn_mult_lr_list = list()
        neck_mult_lr_list = list()
        bbox_head_reg_mult_lr_list = list()
        extra_bbox_head_mult_lr_list = list()
        rpn_head_mult_lr_list = list()
        named_modules = dict(model.named_modules())
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                params.append(param_group)
                continue

            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            elif re.search(r'bbox_head.(\d+).fc_cls.(weight|bias)', name):
                param_group['lr'] = base_lr * bbox_head_lr_mult
                bbox_head_mult_lr_list.append((name, param_group['lr']))
            elif 'conv2.conv_offset' in name:
                param_group['lr'] = base_lr * dcn_lr_mult
                dcn_mult_lr_list.append((name, param_group['lr']))
            elif name.startswith('neck.1.'):
                param_group['lr'] = base_lr * neck_lr_mult
                neck_mult_lr_list.append((name, param_group['lr']))
            elif re.search(r'bbox_head.(\d+).fc_reg.(weight|bias)', name):
                param_group['lr'] = base_lr * bbox_head_reg_lr_mult
                bbox_head_reg_mult_lr_list.append((name, param_group['lr']))
            elif re.search(r'bbox_head.(\d+).extra_fc_(cls|reg).(weight|bias)', name):
                param_group['lr'] = base_lr * extra_bbox_head_lr_mult
                extra_bbox_head_mult_lr_list.append((name, param_group['lr']))
            elif name.startswith('rpn_head.') and not name.startswith('rpn_head.rpn_'):
                param_group['lr'] = base_lr * rpn_head_lr_mult
                rpn_head_mult_lr_list.append((name, param_group['lr']))
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult

            module_name = name.replace('.weight', '').replace('.bias', '')
            if module_name in named_modules and base_wd is not None:
                module = named_modules[module_name]
                # if this Conv2d is depthwise Conv2d
                if isinstance(module, torch.nn.Conv2d) and \
                        module.in_channels == module.groups:
                    param_group['weight_decay'] = base_wd * dwconv_decay_mult
            # otherwise use the global settings

            params.append(param_group)
        if bbox_head_lr_mult != 1.:
            print('bbox_head mult lr: {}'.format(bbox_head_mult_lr_list))
        if dcn_lr_mult != 1.:
            print('dcn mult lr: {}'.format(dcn_mult_lr_list))
        if neck_lr_mult != 1.:
            print('neck mult lr: {}'.format(neck_mult_lr_list))
        if bbox_head_reg_lr_mult != 1.:
            print('bbox_head_reg mult lr: {}'.format(bbox_head_reg_mult_lr_list))
        if extra_bbox_head_lr_mult != 1.:
            print('extra_bbox_head mult lr: {}'.format(extra_bbox_head_mult_lr_list))
        if rpn_head_lr_mult != 1.:
            print('rpn_head mult lr: {}'.format(rpn_head_mult_lr_list))

    optimizer_cfg['params'] = params

    return build_from_cfg(optimizer_cfg, OPTIMIZERS)
