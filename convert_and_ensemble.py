import os
import os.path as osp
import json
import copy
import numpy as np
from ensemble_boxes import *

trs_anno_sub = r'./data/open_subclass_cocostyle.json'
# trs_anno_sub = r'F:\WorkPlace\其他\复现方法文档\RefuseClassify\home\train\train_subclass_cocostyle_val.json'

model_submits = [
    r'./data/base_nms.bbox.json',
    r'./data/base_ft_nms.bbox.json',
    r'./data/base_less_nms.bbox.json',
]

out_submit = r'./submission.json'

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # if isinstance(obj, time):
        #     return obj.__str__()
        # else:
        #     return super(NpEncoder, self).default(obj)

def dump_json_info(js_train):
    print(js_train.keys())
    print(js_train['info'])
    print(js_train['lincenses'])
    print(js_train['images'][0])
    # print(js_train['type'])
    # print(js_train['annotations'][0])
    print(js_train['categories'])

def create_image_id_infos(js_dict, using_str=True):
    image_id_to_infos = dict()
    for _info in js_dict['images']:
        new_info = copy.deepcopy(_info)
        if using_str:
            image_id = new_info['image_id']
        else:
            image_id = new_info['id']
        if image_id_to_infos.__contains__(image_id):
            raise ValueError('duplicate keys: {}'.format(new_info))
        image_id_to_infos[image_id] = new_info
    return image_id_to_infos

def create_image_id_submit_infos(model_submit_files, id_to_str):
    submit_dict_list = list()
    for _submit_file in model_submit_files:
        with open(_submit_file, 'r', encoding='utf-8') as f:
            _submit_infos = json.load(f)
        for _info in _submit_infos:
            _info['image_id'] = id_to_str[_info['image_id']]
        submit_dict = dict()
        for _info in _submit_infos:
            image_id = _info['image_id']
            if not submit_dict.__contains__(image_id):
                submit_dict[image_id] = list()
            submit_dict[image_id].append(_info)
        submit_dict_list.append(submit_dict)
    return submit_dict_list

def create_relation(js_dict):
    id_to_str = dict()
    for _info in js_dict['images']:
        new_info = copy.deepcopy(_info)
        str_imgid = new_info['image_id']
        int_imgid = new_info['id']
        if id_to_str.__contains__(int_imgid):
            raise ValueError('duplicate keys: {}'.format(new_info))
        id_to_str[int_imgid] = str_imgid
    return id_to_str

def norm_bbox(bbox, height, width):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = x1 + bbox[2]
    y2 = y1 + bbox[3]
    return [x1 / width, y1 / height, x2 / width, y2 / height]

def denorm_bbox(bbox, height, width):
    x1 = bbox[0]
    y1 = bbox[1]
    w  = bbox[2] - x1
    h  = bbox[3] - y1
    return [x1 * width, y1 * height, w * width, h * height]

def create_input_struct(image_id_to_infos, submit_dict_list):
    input_dict = dict()
    image_id_list = list(image_id_to_infos.keys())
    for image_id in image_id_list:
        height = image_id_to_infos[image_id]['height']
        width = image_id_to_infos[image_id]['width']
        input_dict[image_id] = [[], [], []]
        for submit_dict in submit_dict_list:
            if not submit_dict.__contains__(image_id):
                continue
            submit_infos = submit_dict[image_id]
            bbox_list = list()
            score_list = list()
            cid_list = list()
            for _info in submit_infos:
                bbox  = _info['bbox']
                normed_bbox = norm_bbox(bbox, height, width)
                bbox_list.append(normed_bbox)
                score_list.append(_info['score'])
                cid_list.append(_info['category_id'])
            input_dict[image_id][0].append(bbox_list)
            input_dict[image_id][1].append(score_list)
            input_dict[image_id][2].append(cid_list)
    return input_dict

if __name__ == '__main__':

    with open(trs_anno_sub, 'r', encoding='utf-8') as f:
        trs_anno_sub_json = json.load(f)
    id_to_str = create_relation(trs_anno_sub_json)
    dump_json_info(trs_anno_sub_json)

    using_str = True
    image_id_to_infos = create_image_id_infos(trs_anno_sub_json, using_str)

    submit_dict_list = create_image_id_submit_infos(model_submits, id_to_str)

    input_struct = create_input_struct(image_id_to_infos, submit_dict_list)
    output_struct = dict()
    for k, v in input_struct.items():
        # boxes, scores, labels = nms_method(v[0], v[1], v[2], method=3, iou_thr=0.9, weights=None)
        boxes, scores, labels = weighted_boxes_fusion(v[0], v[1], v[2], iou_thr=0.7, weights=None, skip_box_thr=0.0)
        # print(boxes, scores, labels)
        output_struct[k] = [boxes, scores, labels]

    output_list = list()
    for k, v in output_struct.items():
        image_id = k
        height = image_id_to_infos[image_id]['height']
        width = image_id_to_infos[image_id]['width']
        bbox_nums = len(v[0])
        for i in range(bbox_nums):
            tmp_dict = dict()
            bbox = v[0][i]
            tmp_dict['image_id'] = image_id
            tmp_dict['category_id'] = int(v[2][i])
            tmp_dict['bbox'] = denorm_bbox(bbox, height, width)
            tmp_dict['score'] = v[1][i]
            output_list.append(tmp_dict)
    print(output_list)
    with open(out_submit, 'w', encoding='utf-8') as f:
        json.dump(output_list, f, cls=MyEncoder)