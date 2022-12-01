import logging
import numpy as np

logger = logging.getLogger(__name__)


def merge_splits_feat(data):
    merged_feat = None
    for split in ['train_data', 'val_data', 'test_data']:
        if hasattr(data, split):
            split_data = getattr(data, split)
            if split_data is not None and 'x' in split_data:
                if merged_feat is None:
                    merged_feat = split_data['x']
                else:
                    merged_feat = \
                        np.concatenate((merged_feat, split_data['x']), axis=0)
    if merged_feat is None:
        raise ValueError('Not support data type for merged feature.')
    return merged_feat


def vfl_binning(feat, num_bins, strategy='uniform'):
    num_features = feat.shape[1]
    bin_edges = np.zeros(num_features, dtype=object)

    for i in range(num_features):
        col = feat[:, i]
        col_min, col_max = np.min(col), np.max(col)
        if col_min == col_max:
            logger.warning(
                f"Feature {i} is constant and will be replaced with 0.")
            bin_edges[i] = np.array([-np.inf, np.inf])
            continue
        if strategy == "uniform":
            bin_edges[i] = np.linspace(col_min, col_max, num_bins[i] + 1)
        elif strategy == "quantile":
            quantiles = np.linspace(0, 100, num_bins[i] + 1)
            bin_edges[i] = np.asarray(np.percentile(col, quantiles))

    return bin_edges
