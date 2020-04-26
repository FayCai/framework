from mmcv.runner import Hook
from mmdet.datasets.pipelines.transforms import GridMask

class GridMaskHook(Hook):

    def __init__(self, dataset):
        # gridmask obj
        self.grid_obj = None
        for obj in dataset[0].pipeline.transforms:
            if isinstance(obj, GridMask):
                self.grid_obj = obj

    def before_train_iter(self, runner):
        cur_iter = runner.iter()
        max_iter = runner.max_iters()
        if self.grid_obj is not None:
            self.grid_obj.set_prob(cur_iter, max_iter)
        