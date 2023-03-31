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

# An example for converting torch training process to FS training process

# Refer to `federatedscope.core.trainers.BaseTrainer` for interface.

# Try with FEMNIST:
#  python federatedscope/main.py --cfg scripts/example_configs/femnist.yaml \
#  trainer.type mytorchtrainer federate.sample_client_rate 0.01 \
#  federate.total_round_num 5 eval.best_res_update_round_wise_key test_loss



def distilllifecycle(lifecycle):
    """
    Manage the lifecycle of the variables within context, \
    and blind these operations from user.

    Arguments:
        lifecycle: the type of lifecycle, choose from "batch/epoch/routine"
    """
    if lifecycle == "routine":

        def decorate(func):
            def wrapper(self, mode, hooks_set, dataset_name=None):
                self.ctx.track_mode(mode)
                self.ctx.track_split(dataset_name or mode)
                self.local_ctx.track_mode(mode)
                self.local_ctx.track_split(dataset_name or mode)

                res = func(self, mode, hooks_set, dataset_name)

                # Clear the variables at the end of lifecycles
                self.ctx.clear(lifecycle)
                self.local_ctx.clear(lifecycle)
                # rollback the model and data_split
                self.ctx.reset_mode()
                self.ctx.reset_split()
                self.local_ctx.reset_mode()
                self.local_ctx.reset_split()
                # Move the model into CPU to avoid memory leak
                self.discharge_model()
                self.discharge_local_model()

                return res

            return wrapper
    else:

        def decorate(func):
            def wrapper(self, *args, **kwargs):
                res = func(self, *args, **kwargs)
                # Clear the variables at the end of lifecycles
                self.ctx.clear(lifecycle)
                self.local_ctx.clear(lifecycle)
                return res

            return wrapper

    return decorate



class MyTorchTrainer(GeneralTorchTrainer):
    def __init__(self, model, data, device, config, local_model=None, only_for_eval=False, monitor=None):
        self._cfg = config
        
        self.ctx = cod_context(model, self.cfg, data, device)
        self.local_ctx = cod_context(local_model, self.cfg, data, device)
        setattr(self.ctx,'if_distill',True)
        # Parse data and setup init vars in ctx
        self._setup_data_related_var_in_ctx(self.ctx)
        self._setup_data_related_var_in_ctx(self.local_ctx)
        assert monitor is not None, \
            f"Monitor not found in trainer with class {type(self)}"
        self.ctx.monitor = monitor
        self.local_ctx.monitor = monitor
        # the "model_nums", and "models" are used for multi-model case and
        # model size calculation
        self.model_nums = 1
        self.ctx.models = [model]
        self.local_ctx.models = [local_model]

        # "mirrored_models": whether the internal multi-models adopt the
        # same architects and almost the same behaviors,
        # which is used to simply the flops, model size calculation
        self.ctx.mirrored_models = False

        # Atomic operation during training/evaluation
        self.hooks_in_train = collections.defaultdict(list)

        # By default, use the same trigger keys
        self.hooks_in_eval = copy.deepcopy(self.hooks_in_train)
        self.hooks_in_ft = copy.deepcopy(self.hooks_in_train)
        self.hooks_in_distill = copy.deepcopy(self.hooks_in_train)

        # register necessary hooks into self.hooks_in_train and
        # self.hooks_in_eval
        if not only_for_eval:
            self.register_default_hooks_train()
        if self.cfg.finetune.before_eval:
            self.register_default_hooks_ft()
        self.register_default_hooks_eval()
        self.register_default_hooks_distill()

        if self.cfg.federate.mode == 'distributed':
            self.print_trainer_meta_info()
        else:
            # in standalone mode, by default, we print the trainer info only
            # once for better logs readability
            pass
        
        category_init_dict = dict()
        self.num_classes = self.cfg.model.out_channels
        cifar_dataset = self.ctx.train_data.dataset.dataset
        self.train_category_subsets = {target: Subset(self.ctx.train_data, [i for i, (x, y) in enumerate(self.ctx.train_data) if y == target]) 
                                        for _, target in cifar_dataset.class_to_idx.items()}
        self.train_category_loaders = {target: (get_dataloader(subset, self.cfg, 'train') if len(subset)>0 else DataLoader(subset))
                    for target, subset in self.train_category_subsets.items()}
        self.test_category_subsets = {target: Subset(self.ctx.test_data, [i for i, (x, y) in enumerate(self.ctx.test_data) if y == target]) 
                                        for _, target in cifar_dataset.class_to_idx.items()}
        self.test_category_loaders = {target: (get_dataloader(subset, self.cfg, 'test') if len(subset)>0 else DataLoader(subset))
                    for target, subset in self.test_category_subsets.items()}
        category_init_dict['num_classes'] = self.num_classes
        category_init_dict['train_category_subsets'] = copy.deepcopy(self.train_category_subsets)
        category_init_dict['train_category_loaders'] = copy.deepcopy(self.train_category_loaders)
        category_init_dict['test_category_subsets'] = copy.deepcopy(self.test_category_subsets)
        category_init_dict['test_category_loaders'] = copy.deepcopy(self.test_category_loaders)

        self.ctx.merge_from_dict(category_init_dict)
        self.local_ctx.merge_from_dict(category_init_dict)

    @property
    def cfg(self):
        return self._cfg
    
    @cfg.setter
    def cfg(self, new_cfg):
        self._cfg = new_cfg
        self.ctx.cfg = new_cfg
        self.local_ctx.cfg = new_cfg
        self._setup_data_related_var_in_ctx(self.ctx)
        self._setup_data_related_var_in_ctx(self.local_ctx)

    def setup_data_for_distill(self, ctx):
        if isinstance(ctx.data, ClientData):
            ctx.data['train'] = ctx.data.train_data
            ctx.data['test'] = ctx.data.test_data
        else:
            logger.warning(f'The data type should be `ClientData` to '
                           f'enable new `config`, but got '
                           f'{type(ctx.data)} instead.')

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
                        init_dict["{}_loader".format(split)] = get_dataloader(data.get(split), self.cfg, split)
                    elif isinstance(data.get(split), DataLoader):
                        init_dict["{}_loader".format(split)] = data.get(split)
                        init_dict["num_{}_data".format(split)] = len(
                            data.get(split).dataset)
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


    def _setup_data_related_var_in_ctx(self, ctx):
        if ctx.get('if_distill') and ctx.if_distill:
            self.setup_data_for_distill(ctx)
        else:
            self.setup_data(ctx)

        init_dict = self.parse_data(ctx.data)

        ctx.merge_from_dict(init_dict)

    def register_default_hooks_distill(self):
        self.register_hook_in_distill(self._hook_on_distill_fit_start_init,
                                      "on_fit_start")                                  
        self.register_hook_in_distill(self._hook_on_distill_epoch_start,
                                      "on_epoch_start")
        self.register_hook_in_distill(self._hook_on_distill_batch_start_init,
                 "on_batch_start")
        self.register_hook_in_distill(self._hook_on_distill_batch_forward,
                 "on_batch_forward")
        self.register_hook_in_distill(self._hook_on_distill_batch_forward_regularizer,
                 "on_batch_forward")
        self.register_hook_in_distill(self._hook_on_distill_batch_backward,
                 "on_batch_backward")
        self.register_hook_in_distill(self._hook_on_distill_batch_end, "on_batch_end")
        self.register_hook_in_distill(self._hook_on_distill_fit_end, "on_fit_end")

    def _hook_on_distill_fit_start_init(self, ctx):
        local_ctx = self.local_ctx
        global_ctx = ctx
        # Initialize optimizer here to avoid the reuse of optimizers
        # across different routines
        for ctx in (local_ctx, global_ctx):
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                ctx.optimizer = get_optimizer(ctx.model,
                                            **ctx.cfg.distill.optimizer)
                ctx.scheduler = get_scheduler(ctx.optimizer,
                                            **ctx.cfg.distill.scheduler)
            ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
            ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
            ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
            ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
            ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)
            
    def _hook_on_distill_epoch_start(self, ctx):
        # prepare dataloader
        # the iterator reset to the first (maybe)
        local_ctx = self.local_ctx
        global_ctx = ctx
        if ctx.cur_mode == "test":
            for ctx in (local_ctx, global_ctx):
                for target in range(self.num_classes):
                    if not isinstance(ctx.get("test_category_loaders").get(target),
                                        ReIterator):
                        ctx["test_category_loaders"][target]=\
                                ReIterator(ctx['test_category_loaders'][target])
                    else:
                        ctx["test_category_loaders"][target].reset()
        else:
            for ctx in (local_ctx, global_ctx):
                for target in range(self.num_classes):
                    if not isinstance(ctx.get("train_category_loaders").get(target),
                                        ReIterator):
                        ctx["train_category_loaders"][target]=\
                                ReIterator(ctx['train_category_loaders'][target])
                    else:
                        ctx["train_category_loaders"][target].reset()
            

    def _hook_on_distill_batch_start_init(self, ctx):
        local_ctx = self.local_ctx
        global_ctx = ctx
        for ctx in (local_ctx, global_ctx):
            try:
                ctx.data_batch = CtxVar(
                    next(ctx.get("{}_category_loaders".format(ctx.cur_split))[ctx.cur_category]),
                    LIFECYCLE.BATCH)
            except StopIteration:
                raise StopIteration

    def _hook_on_distill_batch_forward(self, ctx):
        local_ctx = self.local_ctx
        global_ctx = ctx
        if ctx.cur_mode == 'test':
            for ctx in (local_ctx, global_ctx):           
                x, label = [_.to(ctx.device) for _ in ctx.data_batch]
                pred = ctx.model(x)
                if len(label.size()) == 0:
                    label = label.unsqueeze(0)

                ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
                ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
                ctx.loss_batch = CtxVar(ctx.criterion(pred, label), LIFECYCLE.BATCH)
                ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)
        else:
            student=teacher=None
            if local_ctx.learn_from_the_other[local_ctx.cur_category]:
                student = local_ctx
                teacher = global_ctx
            else:
                student = global_ctx
                teacher = local_ctx

            x, label = [_.to(student.device) for _ in student.data_batch]
            pred_s = student.model(x)
            pred_t = teacher.model(x)
            pred_t = torch.div(pred_t, teacher.cfg.distill.temperature)
            softmaxlayer = torch.nn.Softmax(dim=0)
            pred_t = softmaxlayer(pred_t)
            distilled_pred_s = torch.div(pred_s, student.cfg.distill.temperature)
            if len(label.size()) == 0:
                label = label.unsqueeze(0)
            
            student.y_true = CtxVar(label, LIFECYCLE.BATCH)
            student.y_prob = CtxVar(pred_s, LIFECYCLE.BATCH)
            student.loss_batch = CtxVar(student.criterion(pred_s, label)+\
                student.cfg.distill.alpha*student.criterion(distilled_pred_s, pred_t), LIFECYCLE.BATCH)
            student.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)

    def _hook_on_distill_batch_forward_regularizer(self, ctx):
        if self.ctx.cur_mode == 'test':
            return
        local_ctx = self.local_ctx
        global_ctx = ctx
        if local_ctx.learn_from_the_other[local_ctx.cur_category]:
            ctx = local_ctx
        else:
            ctx = global_ctx        
        ctx.loss_regular = CtxVar(
            self.cfg.regularizer.mu * ctx.regularizer(ctx), LIFECYCLE.BATCH)
        ctx.loss_task = CtxVar(ctx.loss_batch + ctx.loss_regular,
                            LIFECYCLE.BATCH)

    def _hook_on_distill_batch_backward(self, ctx):
        if self.ctx.cur_mode == 'test':
            return
        local_ctx = self.local_ctx
        global_ctx = ctx
        if local_ctx.learn_from_the_other[local_ctx.cur_category]:
            ctx = local_ctx
        else:
            ctx = global_ctx

        ctx.optimizer.zero_grad()
        ctx.loss_task.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                        ctx.grad_clip)

        ctx.optimizer.step()
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def _hook_on_distill_batch_end(self, ctx):
        local_ctx = self.local_ctx
        global_ctx = ctx
        if ctx.cur_mode == 'train':
            if local_ctx.learn_from_the_other[local_ctx.cur_category]:
                ctx = local_ctx
            else:
                ctx = global_ctx
            
            for ctxx in (local_ctx, global_ctx):
            # update statistics
                ctxx.num_samples += ctx.batch_size
                ctxx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
                ctxx.loss_regular_total += float(ctx.get("loss_regular", 0.))
                # cache label for evaluate
                ctxx.ys_true.append(ctx.y_true.detach().cpu().numpy())
                ctxx.ys_prob.append(ctx.y_prob.detach().cpu().numpy())
        else:
            for ctx in (local_ctx, global_ctx):
                ctx.num_samples += ctx.batch_size
                ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
                ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
                # cache label for evaluate
                ctx.ys_true.append(ctx.y_true.detach().cpu().numpy())
                ctx.ys_prob.append(ctx.y_prob.detach().cpu().numpy())


    def _hook_on_distill_fit_end(self, ctx):
        local_ctx = self.local_ctx
        global_ctx = ctx
        if ctx.cur_mode == "train":
            for ctx in (local_ctx, global_ctx):
                ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
                ctx.ys_prob = CtxVar(np.concatenate(ctx.ys_prob), LIFECYCLE.ROUTINE)
                results = ctx.monitor.eval(ctx)
                setattr(ctx, 'eval_metrics', results)
        else:
            classes = self.num_classes
            #correct_list = [[0. for _ in range(classes)] for x in range(y_true.shape[1])]
            for idx, ctx in enumerate((local_ctx, global_ctx)):
                y_true = np.concatenate(ctx.ys_true)
                y_prob = np.concatenate(ctx.ys_prob)
                y_pred = np.argmax(y_prob, axis=1)
                is_class = [np.where(y_true[:] == k)[0] for k in range(classes)]
                correct = [y_true[is_class[k]] == y_pred[is_class[k]] for k in range(classes)]
                for k in range(classes):
                    if isinstance(correct[k], bool):
                        correct[k]=np.array([correct[k]])
                acc_list = [float(np.sum(correct[k]))/len(correct[k]) if len(correct[k])!=0 else 0 for k in range(classes)]
                acc_list = np.array(acc_list)
                if idx == 0:
                    setattr(local_ctx, 'acc_list', acc_list)
                else :
                    setattr(global_ctx, 'acc_list', acc_list)
                setattr(ctx, 'eval_metrics', {})
            learn_from_the_other = local_ctx.acc_list < global_ctx.acc_list
            logger.info("#### distill eval ####")
            logger.info(local_ctx.acc_list)
            logger.info(global_ctx.acc_list)
            logger.info(learn_from_the_other)
            setattr(local_ctx, 'learn_from_the_other', learn_from_the_other)

    def register_hook_in_distill(self,
                                 new_hook,
                                 trigger,
                                 insert_pos=None,
                                 base_hook=None,
                                 insert_mode="before"):
        hooks_dict = self.hooks_in_distill
        self._register_hook(base_hook, hooks_dict, insert_mode, insert_pos,
                            new_hook, trigger)

    def update(self, model_parameters, strict=False):
        """
            Called by the FL client to update the model parameters
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        for key in model_parameters:
            if key.startswith('fc'):
                continue
            model_parameters[key] = param2tensor(model_parameters[key])
        # Due to lazy load, we merge two state dict
        merged_param = merge_param_dict(self.ctx.model.state_dict().copy(),
                                        self._param_filter(model_parameters))
        self.ctx.model.load_state_dict(merged_param, strict=strict)


    def distill(self, target_data_split_name="train", hooks_set=None, eval_before=False):
        hooks_set = hooks_set or self.hooks_in_distill

        self.ctx.check_split(target_data_split_name)
        self.local_ctx.check_split(target_data_split_name)
        if eval_before:
            self.ctx.model.eval()
            self.local_ctx.model.eval()
            num_samples = self._run_distill_routine(MODE.TEST, hooks_set, "test")
            self.ctx.model.train()
            self.local_ctx.model.train()
        else:
            num_samples = self._run_distill_routine(MODE.TRAIN, hooks_set, target_data_split_name)

        return num_samples, self.ctx.model.cpu().state_dict(), self.ctx.eval_metrics

    def evaluate(self, target_data_split_name="test"):
        self.ctx.if_distill=False
        self._setup_data_related_var_in_ctx(self.ctx)
        with torch.no_grad():
            super(GeneralTorchTrainer, self).evaluate(target_data_split_name)
        self.ctx.if_distill=True
        return self.ctx.eval_metrics
    
    @distilllifecycle(LIFECYCLE.ROUTINE)
    def _run_distill_routine(self, mode, hooks_set, dataset_name=None):
        for hook in hooks_set["on_fit_start"]:
            hook(self.ctx) 

        target_list = [i for i in range(self.num_classes)]
        if self.cfg.distill.shuffle and self.ctx.cur_mode == 'train':
            random.shuffle(target_list)
        logger.info(target_list)
        for target in target_list:
            if self.ctx.cur_mode == 'train' and len(self.train_category_subsets[target])==0:
                continue
            self.ctx.cur_category= CtxVar(target, "routine")
            self.local_ctx.cur_category = CtxVar(target, "routine")
            self._run_distill_epoch(hooks_set)

        for hook in hooks_set["on_fit_end"]:
            hook(self.ctx)
        return self.ctx.num_samples

    @distilllifecycle(LIFECYCLE.EPOCH)
    def _run_distill_epoch(self, hooks_set):
        for epoch_i in range(
                getattr(self.ctx, f'num_distill_{self.ctx.cur_split}_epoch')[getattr(self.ctx, 'cur_category')]):
            self.ctx.cur_epoch_i = CtxVar(epoch_i, "epoch")
            self.local_ctx.cur_epoch_i = CtxVar(epoch_i, "epoch")
            
            for hook in hooks_set["on_epoch_start"]:
                hook(self.ctx)
            
            self._run_distill_batch(hooks_set)

            for hook in hooks_set["on_epoch_end"]:
                hook(self.ctx)

    @distilllifecycle(LIFECYCLE.BATCH)
    def _run_distill_batch(self, hooks_set):
        a = getattr(self.ctx, f'num_distill_{self.ctx.cur_split}_batch')[getattr(self.ctx, 'cur_category')]
        logger.info(f"target: {self.ctx.cur_category} batch_num: {a}")
        for batch_i in range(getattr(self.ctx, f'num_distill_{self.ctx.cur_split}_batch')[getattr(self.ctx, 'cur_category')]):
            
            self.ctx.cur_batch_i = CtxVar(batch_i, LIFECYCLE.BATCH)
            self.local_ctx.cur_batch_i = CtxVar(batch_i, LIFECYCLE.BATCH)

            for hook in hooks_set["on_batch_start"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_forward"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_backward"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_end"]:
                hook(self.ctx)
        

    def discharge_local_model(self):
        if not self.cfg.federate.share_local_model:
            if torch is None:
                pass
            else:
                self.local_ctx.model.to(torch.device("cpu"))



def call_my_torch_trainer(trainer_type):
    if trainer_type == 'cod_trainer':
        return MyTorchTrainer


register_trainer('cod_trainer', call_my_torch_trainer)
