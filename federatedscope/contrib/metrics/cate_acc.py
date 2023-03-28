from federatedscope.register import register_metric
import numpy as np



def load_cateacc_metrics(ctx, y_true, y_prob, y_pred, **kwargs):
    if ctx.cur_split =='test':
        classes = 10
        #correct_list = [[0. for _ in range(classes)] for x in range(y_true.shape[1])]
        acc_list =[]
        for i in range(y_true.shape[1]):
            is_class = [np.where(y_true[:, i] == k)[0] for k in range(classes)]
            correct = [y_true[is_class[k], i] == y_pred[is_class[k], i] for k in range(classes)]
            acc_list.append([float(np.sum(correct[k]))/len(correct[k]) if len(correct[k])!=0 else 0 for k in range(classes)])
        print(acc_list)
        acc_list = np.array(acc_list)
        results = [np.sum(acc_list[:,k])/ len(acc_list[:,k]) for k in range(classes)]
    else:
        results = None
    return results





def call_cateacc_metric(types):
    if 'cate_acc_no_merge' in types:
        the_larger_the_better = True
        return 'cate_acc_no_merge', load_cateacc_metrics, the_larger_the_better


register_metric('cate_acc_no_merge', call_cateacc_metric)
