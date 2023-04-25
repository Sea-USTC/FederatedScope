from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.data.wrap_dataset import WrapDataset
from federatedscope.core.data.base_data import ClientData

from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator
from federatedscope.core.auxiliaries.utils import param2tensor, \
    merge_param_dict
from federatedscope.contrib.trainer.cod_context import cod_context

import random
import collections
import copy
import numpy as np
import logging

try:
    import torch
    from torch.utils.data import DataLoader, Subset, Dataset
except ImportError:
    torch = None
    DataLoader = None
    Subset =None
    Dataset = None
import copy


logger = logging.getLogger(__name__)


class RodTorchTrainer(GeneralTorchTrainer):
    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        category_init_dict = dict()
        self.num_classes = self.cfg.model.out_channels
        self.train_category_subsets = {target: Subset(self.ctx.train_data, [i for i, (x, y) in enumerate(self.ctx.train_data) if y == target]) 
                                        for target in range(self.num_classes)}
        self.sample_per_class=torch.zeros(self.num_classes)
        for target, subset in self.train_category_subsets.items():
            self.sample_per_class[target]=len(subset)
        self.p_head = torch.nn.Sequential(torch.nn.Linear(512, self.num_classes))

        category_init_dict['num_classes'] = self.num_classes
        category_init_dict['sample_per_class'] = self.sample_per_class
        category_init_dict['p_head'] = self.p_head

        self.ctx.merge_from_dict(category_init_dict)

    def parse_data(self, data):
        """Populate "${split}_data", "${split}_loader" and "num_${
        split}_data" for different data splits
        """
        init_dict = dict()
        if isinstance(data, dict):
            for split in data.keys():
                if split not in ['train', 'val', 'test']:
                    continue
                init_dict["{}_data".format(split)] = None
                init_dict["{}_loader".format(split)] = None
                init_dict["num_{}_data".format(split)] = 0
                if data.get(split, None) is not None:
                    if isinstance(data.get(split), Dataset):
                        init_dict["{}_data".format(split)] = data.get(split)
                        init_dict["num_{}_data".format(split)] = len(
                            data.get(split))
                    elif isinstance(data.get(split), DataLoader):
                        init_dict["{}_loader".format(split)] = data.get(split)
                        init_dict["num_{}_data".format(split)] = len(
                            data.get(split).dataset)
                        init_dict["{}_data".format(split)] = data.get(split).dataset
                    elif isinstance(data.get(split), dict):
                        init_dict["{}_data".format(split)] = data.get(split)
                        init_dict["num_{}_data".format(split)] = len(
                            data.get(split)['y'])
                    else:
                        raise TypeError("Type {} is not supported.".format(
                            type(data.get(split))))
        else:
            raise TypeError("Type of data should be dict.")
        return init_dict


    def _hook_on_fit_start_init(self, ctx):
        # prepare model and optimizer
        ctx.model.to(ctx.device)
        ctx.p_head.to(ctx.device)
        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            # Initialize optimizer here to avoid the reuse of optimizers
            # across different routines
            ctx.optimizer = torch.optim.SGD(
                [{'params': ctx.model.parameters()},{'params': ctx.p_head.parameters()}],
                lr=ctx.cfg[ctx.cur_mode].optimizer.lr)
            ctx.scheduler = get_scheduler(ctx.optimizer,
                                          **ctx.cfg[ctx.cur_mode].scheduler)
        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)
        if ctx.cur_mode in [MODE.TEST, MODE.VAL]:
            ctx.ys_prob_personal = CtxVar([], LIFECYCLE.ROUTINE)   
        
    def _hook_on_batch_forward(self, ctx):
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]
        pred, feature = ctx.model(x)
        empirical_pred = ctx.p_head(feature)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(
            ctx.criterion(empirical_pred, label)+BalancedSoftmaxRisk(pred, label, self.sample_per_class),
            LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)
        if ctx.cur_mode in [MODE.TEST, MODE.VAL]:
            ctx.y_prob_personal = CtxVar(empirical_pred, LIFECYCLE.BATCH)
            #ctx.loss_batch_global = CtxVar(ctx.criterion(pred, label), LIFECYCLE.BATCH)
            #ctx.loss_batch_personal = CtxVar(ctx.criterion(empirical_pred, label), LIFECYCLE.BATCH)

    def _hook_on_batch_end(self, ctx):
        # update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        # cache label for evaluate
        ctx.ys_true.append(ctx.y_true.detach().cpu().numpy())
        ctx.ys_prob.append(ctx.y_prob.detach().cpu().numpy())
        if ctx.cur_mode in [MODE.TEST, MODE.VAL]:
            ctx.ys_prob_personal.append(ctx.y_prob_personal.detach().cpu().numpy())


    def _hook_on_fit_end(self, ctx):
        ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar(np.concatenate(ctx.ys_prob), LIFECYCLE.ROUTINE)
        results = ctx.monitor.eval(ctx)
        setattr(ctx, 'eval_metrics', results)
        if ctx.cur_mode in [MODE.TEST, MODE.VAL]:
            ctx.ys_prob_personal = CtxVar(np.concatenate(ctx.ys_prob_personal), LIFECYCLE.ROUTINE)
            ctx.ys_prob = ctx.ys_prob_personal
            results = ctx.monitor.eval(ctx)
            setattr(ctx, 'eval_metrics_personal', results)


    #@use_diff
    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)

        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)

        return num_samples, self.get_model_para(), self.ctx.eval_metrics

    def evaluate(self, target_data_split_name="test", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_eval

        if self.ctx.check_split(target_data_split_name, skip=True):
            self.ctx.model.eval()
            self._run_routine(MODE.TEST, hooks_set, target_data_split_name)
            self.ctx.model.train()
        else:
            self.ctx.eval_metrics = dict()
            self.ctx.eval_metrics_personal = dict()

        return self.ctx.eval_metrics, self.ctx.eval_metrics_personal
    
    def discharge_model(self):
        """
        Discharge the model from GPU device
        """
        # Avoid memory leak
        if not self.cfg.federate.share_local_model:
            if torch is None:
                pass
            else:
                self.ctx.model.to(torch.device("cpu"))
                self.ctx.p_head.to(torch.device("cpu"))

# https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification
def BalancedSoftmaxRisk(logits, label, sample_per_class):
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = torch.nn.functional.cross_entropy(input=logits, target=label)
    return loss


def call_my_torch_trainer(trainer_type):
    if trainer_type == 'rod_trainer':
        return RodTorchTrainer


register_trainer('rod_trainer', call_my_torch_trainer)