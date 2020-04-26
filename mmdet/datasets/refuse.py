import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.core import eval_recalls
from mmdet.utils import print_log
from .custom import CustomDataset
from .coco import CocoDataset
from .registry import DATASETS

from terminaltables import AsciiTable
import itertools
import json

@DATASETS.register_module
class RefuseDataset(CocoDataset):

    CLASSES = ('瓜子壳', '核桃', '花生壳', '西瓜子', '枣核', '话梅核', '西瓜皮', '香蕉皮',
                '荔枝壳', '苹果核', '干果', '饼干', '面包', '糖果', '风干食品', '糕饼', '薯片',
                '树叶', '杂草', '绿植', '豆类', '动物内脏', '贝类去硬壳', '面食', '排骨-小肋排', 
                '鸡骨头', '螺蛳', '菜叶', '菌菇类', '粽子', '小龙虾', '蟹壳', '中药材', '中药渣',
                '镜子', '玻璃制品', '窗玻璃', '碎玻璃片', '食品及日用品玻璃瓶罐', '保温杯',
                '图书期刊', '报纸', '食品外包装盒', '利乐包', '广告单', '打印纸', '日历', '快递纸袋', 
                '信封', '烟盒', '易拉罐', '金属制品', '吸铁石', '金属瓶罐', '金属工具', '罐头盒', 
                '勺子', '菜刀', '叉子', '金属筷子', '数据线', '塑料玩具', '矿泉水瓶', '塑料泡沫', 
                '塑料包装', '硬塑料', '一次性塑料餐盒餐具', '电线', '塑料衣架', '插座', '篮球', 
                '足球', '食品塑料盒', '食用油桶', '鞋', '书包', '床上用品', '棉被', '丝绸手绢', 
                '枕头', '皮带', '电路板', '充电宝', '木制品', '优盘', '灯管灯泡', '节能灯', '二极管', 
                '纽扣电池', '手机电池', '锂电池', '蓄电池', '胶卷', '指甲油瓶', 'X光片', '农药瓶', 
                '杀虫剂及罐', '墨盒', '染发剂壳', '消毒液瓶', '油漆桶', '药品包装', '药瓶', '废弃针管', 
                '输液管', '口服液瓶', '眼药水瓶', '水银温度计', '胶囊', '药片', '甘蔗皮', '坚果壳', 
                '橡皮泥', '毛发', '棉签', '创可贴', '口红', '笔', '纸巾', '胶带', '湿巾', '水彩笔', 
                '打火机', '防碎气泡膜', '榴莲壳', '睫毛膏', '眼影', '大骨棒', '旧毛巾', '竹制品', 
                '粉笔', '一次性口罩', '一次性手套', '粉底液', '尼龙制品', '尿片', '牙膏皮', '狗尿布', 
                '粉扑', '卫生纸', '烟头', '牙刷', '玉米棒')

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0] + 1,
            _bbox[3] - _bbox[1] + 1,
        ]

    def _proposal2json(self, results):
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = '{}.{}.json'.format(outfile_prefix, 'bbox')
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'bbox')
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = '{}.{}.json'.format(outfile_prefix, 'bbox')
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'bbox')
            result_files['segm'] = '{}.{}.json'.format(outfile_prefix, 'segm')
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'proposal')
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.getAnnIds(imgIds=self.img_ids[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        """Evaluation in COCO protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str: float]
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(metric))

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = {}
        cocoGt = self.coco
        for metric in metrics:
            msg = 'Evaluating {}...'.format(metric)
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
                    log_msg.append('\nAR@{}\t{:.4f}'.format(num, ar[i]))
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError('{} is not in results'.format(metric))
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = self.img_ids
            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.params.maxDets = list(proposal_nums)
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
                    'AR_l@1000'
                ]
                for i, item in enumerate(metric_items):
                    val = float('{:.3f}'.format(cocoEval.stats[i + 6]))
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    ret_dict = dict()
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/blob/03064eb5bafe4a3e5750cc7a16672daf5afe8435/detectron2/evaluation/coco_evaluation.py#L259-L283 # noqa
                    precisions = cocoEval.eval['precision']
                    catIds = cocoGt.getCatIds()
                    # precision has dims (iou, recall, cls, area range, max dets)
                    assert len(catIds) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(catIds):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = cocoGt.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        ap = np.mean(precision) if precision.size else float('nan')
                        results_per_category.append(
                            ('{}'.format(nm['id']), '{}'.format(nm['name']),
                            '{:0.3f}'.format(float(ap * 100))))
                        ret_dict[nm['id']] = [nm['name'], float(ap * 100)]

                    N_COLS = min(9, len(results_per_category) * 3)
                    results_flatten = list(itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (N_COLS // 2)
                    results_2d = itertools.zip_longest(
                        *[results_flatten[i::N_COLS] for i in range(N_COLS)])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print(table.table)
                    with open('debug.json', 'w', encoding='utf-8') as f:
                        json.dump(ret_dict, f)
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]
                for i in range(len(metric_items)):
                    key = '{}_{}'.format(metric, metric_items[i])
                    val = float('{:.3f}'.format(cocoEval.stats[i]))
                    eval_results[key] = val
                eval_results['{}_mAP_copypaste'.format(metric)] = (
                    '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def evaluate_json(self,
                    json_file,
                    metric='bbox',
                    logger=None,
                    classwise=False,
                    proposal_nums=(100, 300, 1000),
                    iou_thrs=np.arange(0.5, 0.96, 0.05)):
        """Evaluation in COCO protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str: float]
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(metric))

        result_files = dict()
        result_files['bbox'] = json_file

        eval_results = {}
        cocoGt = self.coco
        for metric in metrics:
            msg = 'Evaluating {}...'.format(metric)
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
                    log_msg.append('\nAR@{}\t{:.4f}'.format(num, ar[i]))
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError('{} is not in results'.format(metric))
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = self.img_ids
            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.params.maxDets = list(proposal_nums)
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
                    'AR_l@1000'
                ]
                for i, item in enumerate(metric_items):
                    val = float('{:.3f}'.format(cocoEval.stats[i + 6]))
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    ret_dict = dict()
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/blob/03064eb5bafe4a3e5750cc7a16672daf5afe8435/detectron2/evaluation/coco_evaluation.py#L259-L283 # noqa
                    precisions = cocoEval.eval['precision']
                    catIds = cocoGt.getCatIds()
                    # precision has dims (iou, recall, cls, area range, max dets)
                    assert len(catIds) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(catIds):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = cocoGt.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        ap = np.mean(precision) if precision.size else float('nan')
                        results_per_category.append(
                            ('{}'.format(nm['id']), '{}'.format(nm['name']),
                            '{:0.3f}'.format(float(ap * 100))))
                        ret_dict[nm['id']] = [nm['name'], float(ap * 100)]

                    N_COLS = min(9, len(results_per_category) * 3)
                    results_flatten = list(itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (N_COLS // 2)
                    results_2d = itertools.zip_longest(
                        *[results_flatten[i::N_COLS] for i in range(N_COLS)])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print(table.table)
                    with open('debug.json', 'w', encoding='utf-8') as f:
                        json.dump(ret_dict, f)
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]
                for i in range(len(metric_items)):
                    key = '{}_{}'.format(metric, metric_items[i])
                    val = float('{:.3f}'.format(cocoEval.stats[i]))
                    eval_results[key] = val
                eval_results['{}_mAP_copypaste'.format(metric)] = (
                    '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        return eval_results
