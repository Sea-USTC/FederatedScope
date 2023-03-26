from federatedscope.core.trainers.context import Context, CtxVar, LifecycleDict
from federatedscope.core.trainers.utils import calculate_batch_epoch_num
import logging

try:
    import torch

except ImportError:
    torch = None


logger = logging.getLogger(__name__)

class cod_context(Context):
    
    @property
    def num_distill_train_batch(self):
        if self.get("num_distill_train_batch"):
            return self.get("num_distill_train_batch")
        return self._calculate_distill_batch_epoch_num(mode='train')[0]
    
    @property
    def num_distill_train_epoch(self):
        if self.get('num_distill_train_epoch'):
            return self.get('num_distill_train_epoch')
        return self._calculate_distill_batch_epoch_num(mode='train')[2]
    
    @property
    def num_distill_test_batch(self):
        if self.get('num_distill_test_batch'):
            return self.get('num_distill_test_batch')
        return self._calculate_distill_batch_epoch_num(mode='test')[0]
    
    @property
    def num_distill_test_epoch(self):
        if self.get('num_distill_test_epoch'):
            return self.get('num_distill_test_epoch')
        return self._calculate_distill_batch_epoch_num(mode='test')[2]

    def _calculate_distill_batch_epoch_num(self, mode="train"):
        if self.cur_mode is not None and self.cur_mode != mode:
            logger.warning(
                f'cur_mode `{self.cur_mode}` mismatch mode `{mode}`, '
                f'will use `{mode}` to calculate `ctx.var`.')
        if self.cur_split is None:
            logger.warning(
                f'cur_split `{self.cur_split}` not found in data_split, '
                f'will use `train` split to calculate `ctx.var`.')
            cur_split = 'train'
        else:
            cur_split = self.cur_split

        num_batch=dict()
        num_batch_last_epoch=dict()
        num_epoch=dict()
        num_total_batch = dict()
        for target in range(self.num_classes):
            num_batch_last_epoch[target], num_total_batch[target] = None, None
            if mode in ['train']:
                if len(self.train_category_subsets[target])==0:
                    num_epoch[target] = 1
                    num_batch[target] = 0
                    continue
                num_batch[target], num_batch_last_epoch[target], \
                num_epoch[target], num_total_batch[target] = \
                    calculate_batch_epoch_num(
                        self.cfg.distill.num_epoches,
                        "epoch",
                        len(self.train_category_subsets[target]),
                        self.cfg.distill.batchsize,
                        self.cfg.dataloader.drop_last)
            elif mode in ['test']:
                num_epoch[target] = 1
                num_batch[target] = len(self.test_category_subsets[target]) // self.cfg.distill.batchsize + int(
                                        not self.cfg.dataloader.drop_last
                                        and bool(
                                            len(self.test_category_subsets[target]) %
                                            self.cfg.distill.batchsize))
            else:
                raise ValueError(f'Invalid mode {mode}.')

        return num_batch, num_batch_last_epoch, num_epoch, num_total_batch