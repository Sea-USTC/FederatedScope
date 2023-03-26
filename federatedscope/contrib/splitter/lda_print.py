from federatedscope.register import register_splitter
from federatedscope.core.splitters import BaseSplitter


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection





class LDA_printSplitter(BaseSplitter):
    def __init__(self, client_num, alpha=0.5, lowerlimit=1, dir="/mnt/petrelfs/lisiyi/project/1.png"):
        self.alpha = alpha
        self.lowerlimit=lowerlimit
        self.cnt=0
        self.dir=dir
        super(LDA_printSplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset

        tmp_dataset = [ds for ds in dataset]
        label = np.array([y for x, y in tmp_dataset])
        idx_slice = self.dirichlet_distribution_noniid_slice(label,
                                                        self.client_num,
                                                        self.alpha,
                                                        self.lowerlimit,
                                                        prior=prior)
        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        self.cnt+=1
        return data_list

    def _split_according_to_prior(self, label, client_num, prior):
        assert client_num == len(prior)
        classes = len(np.unique(label))
        assert classes == len(np.unique(np.concatenate(prior, 0)))

        # counting
        frequency = np.zeros(shape=(client_num, classes))
        for idx, client_prior in enumerate(prior):
            for each in client_prior:
                frequency[idx][each] += 1
        if self.cnt==2:
            self.heatmapplot(frequency, client_num, classes)
        sum_frequency = np.sum(frequency, axis=0)

        idx_slice = [[] for _ in range(client_num)]
        for k in range(classes):
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            nums_k = np.ceil(frequency[:, k] / sum_frequency[k] *
                            len(idx_k)).astype(int)
            while len(idx_k) < np.sum(nums_k):
                random_client = np.random.choice(range(client_num))
                if nums_k[random_client] > 0:
                    nums_k[random_client] -= 1
            assert len(idx_k) == np.sum(nums_k)
            idx_slice = [
                idx_j + idx.tolist() for idx_j, idx in zip(
                    idx_slice, np.split(idx_k,
                                        np.cumsum(nums_k)[:-1]))
            ]

        for i in range(len(idx_slice)):
            np.random.shuffle(idx_slice[i])
        return idx_slice


    def dirichlet_distribution_noniid_slice(self,
                                            label,
                                            client_num,
                                            alpha,
                                            min_size=1,
                                            prior=None):
        r"""Get sample index list for each client from the Dirichlet distribution.
        https://github.com/FedML-AI/FedML/blob/master/fedml_core/non_iid
        partition/noniid_partition.py

        Arguments:
            label (np.array): Label list to be split.
            client_num (int): Split label into client_num parts.
            alpha (float): alpha of LDA.
            min_size (int): min number of sample in each client
        Returns:
            idx_slice (List): List of splited label index slice.
        """
        if len(label.shape) != 1:
            raise ValueError('Only support single-label tasks!')

        if prior is not None:
            return self._split_according_to_prior(label, client_num, prior)

        num = len(label)
        classes = len(np.unique(label))
        assert num > client_num * min_size, f'The number of sample should be ' \
                                            f'greater than' \
                                            f' {client_num * min_size}.'
        size = 0
        while size < min_size:
            idx_slice = [[] for _ in range(client_num)]
            for k in range(classes):
                # for label k
                idx_k = np.where(label == k)[0]
                np.random.shuffle(idx_k)
                prop = np.random.dirichlet(np.repeat(alpha, client_num))
                # prop = np.array([
                #    p * (len(idx_j) < num / client_num)
                #    for p, idx_j in zip(prop, idx_slice)
                # ])
                # prop = prop / sum(prop)
                prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
                idx_slice = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))
                ]
                size = min([len(idx_j) for idx_j in idx_slice])
        for i in range(client_num):
            np.random.shuffle(idx_slice[i])
        return idx_slice   
     
    def heatmapplot(self, frequency, client_num, classes):
        ylabels = [str(_) for _ in range(1, client_num+1)]
        xlabels = [str(_) for _ in range(1, classes+1)]

        x, y = np.meshgrid(np.arange(classes), np.arange(client_num))
        c = np.random.rand(client_num, classes)-0.5
        red = np.zeros((client_num, classes))-0.4
        
        fig, ax = plt.subplots()
        frequency = np.sqrt(frequency)
        R = frequency/frequency.max()/2
        circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
        col = PatchCollection(circles, array=red.flatten(), cmap="RdYlGn")
        ax.add_collection(col)

        ax.set(xticks=np.arange(classes), yticks=np.arange(client_num),
            xticklabels=xlabels, yticklabels=ylabels)
        ax.set_xticks(np.arange(classes+1)-0.5, minor=True)
        ax.set_yticks(np.arange(client_num+1)-0.5, minor=True)
        ax.set_xlabel("class #")
        ax.set_ylabel("client #")
        ax.grid(which='minor')
        fig.set_size_inches(14,14)
        #fig.colorbar(col)
        fig.savefig(self.dir, dpi=300 )


def call_my_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'lda_print':
        splitter = LDA_printSplitter(client_num, **kwargs)
        return splitter


register_splitter('lda_print', call_my_splitter)