from federatedscope.register import register_splitter
from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.utils import \
    dirichlet_distribution_noniid_slice

import numpy as np


class LDA_lowerSplitter(BaseSplitter):
    def __init__(self, client_num, alpha=0.5, lowerlimit=1):
        self.alpha = alpha
        self.lowerlimit=lowerlimit
        super(LDA_lowerSplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset

        tmp_dataset = [ds for ds in dataset]
        label = np.array([y for x, y in tmp_dataset])
        idx_slice = dirichlet_distribution_noniid_slice(label,
                                                        self.client_num,
                                                        self.alpha,
                                                        self.lowerlimit,
                                                        prior=prior)
        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list


def call_my_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'lda_lowerlimit':
        splitter = LDA_lowerSplitter(client_num, **kwargs)
        return splitter


register_splitter('lda_lowerlimit', call_my_splitter)