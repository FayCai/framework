import numpy as np
import os
import os.path as osp
import json
import copy

# ori_anno = r'F:\data\RefuseClassify\complex_open_data\train\train.json'
ori_anno = r'/home/jovyan/data/2020-Haihua-AI-Challenge·Waste-Sorting-Task-2_105/open.json'

trs_anno = r'./data/open_cocostyle.json'

trs_anno_sub = r'./data/open_subclass_cocostyle.json'

exmp_anno_sub = r'./data/val_open_subclass_cocostyle.json'

def dump_json_info(js_train):
    print(js_train.keys())
    print(js_train['info'])
    print(js_train['lincenses'])
    print(js_train['images'][0])
    # print(js_train['type'])
    # print(js_train['annotations'][0])
    print(js_train['categories'])

def trans_json(ori_dict, trs_dict):
    str_to_id = dict()
    id_count = 1
    for _info in ori_dict['images']:
        new_info = copy.deepcopy(_info)
        str_imgid = new_info['image_id']
        if str_to_id.__contains__(str_imgid):
            raise ValueError('duplicate keys: {}'.format(new_info))
        str_to_id[str_imgid] = id_count
        new_info['file_name'] = new_info['file_name'].split('/')[1]
        new_info['coco_url'] = ''
        new_info['data_captured'] = ''
        new_info['flickr_url'] = ''
        new_info['license'] = 0
        new_info['id'] = str_to_id[str_imgid]
        # del new_info['image_id']
        id_count += 1
        trs_dict['images'].append(new_info)

    for _info in ori_dict['categories']:
        new_info = copy.deepcopy(_info)
        new_info['supercategory'] = ''
        trs_dict['categories'].append(new_info)

def find_subclass(trs_dict):
    with open(exmp_anno_sub, 'r', encoding='utf-8') as f:
        exmp_sub_json = json.load(f)

    # check valid
    cid_name = dict()
    for _info in exmp_sub_json['categories']:
        cid_name[_info['id']] = _info['name']
    for _info in trs_dict['categories']:
        if _info['id'] in cid_name.keys():
            assert cid_name[_info['id']] == _info['name']

    ret_dict = copy.deepcopy(trs_dict)
    ret_dict['categories'] = exmp_sub_json['categories']

    return ret_dict

if __name__ == '__main__':
    with open(ori_anno, 'r', encoding='utf-8') as f:
        # s = f.read()
        # js_train = json.loads(json.dumps(eval(s)))
        js_train = json.load(f)

    dump_json_info(js_train)
    trs_json_train = dict()
    trs_json_train['info'] = js_train['info']
    trs_json_train['lincenses'] = js_train['lincenses']
    trs_json_train['images'] = list()
    # trs_json_train['annotations'] = list()
    trs_json_train['categories'] = list()
    trans_json(js_train, trs_json_train)

    dump_json_info(trs_json_train)
    with open(trs_anno, 'w', encoding='utf-8') as f:
        json.dump(trs_json_train, f)

    trs_json_train_sub = find_subclass(trs_json_train)
    dump_json_info(trs_json_train_sub)
    with open(trs_anno_sub, 'w', encoding='utf-8') as f:
        json.dump(trs_json_train_sub, f)