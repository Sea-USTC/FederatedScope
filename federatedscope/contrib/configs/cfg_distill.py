from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

def extend_distill_cfg(cfg):
    cfg.distill = CN()
    cfg.distill.use = True
    cfg.distill.batchsize = 32
    cfg.distill.temperature = 1
    cfg.distill.alpha=1
    cfg.distill.num_epoches = 3
    cfg.distill.local_train_epoches = 20
    cfg.distill.shuffle = True

    cfg.distill.optimizer = CN(new_allowed=True)
    cfg.distill.optimizer.type = 'SGD'
    cfg.distill.optimizer.lr = 0.0001
    
    cfg.distill.scheduler = CN(new_allowed=True)
    cfg.distill.scheduler.type = ''
    cfg.distill.scheduler.warmup_ratio = 0.0


register_config("distill", extend_distill_cfg)