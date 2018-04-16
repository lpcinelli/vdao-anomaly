import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from scipy import interp
from sklearn.metrics import auc, roc_curve

EPS = 1e-8

def _sanitize(y_true, y_pred, threshold, typecast='float32'):
    y_true = K.cast(y_true, typecast)
    y_pred = K.cast(y_pred > threshold, typecast)
    return y_true, y_pred


def _tn(y_true, y_pred, typecast='float32'):
    good_preds = K.cast(K.equal(y_pred, y_true), typecast)
    true_neg = K.cast(
        K.sum(good_preds * K.cast(K.equal(y_true,0), typecast)), typecast)
    return true_neg


def _tp(y_true, y_pred, typecast='float32'):
    good_preds = K.cast(K.equal(y_pred, y_true), typecast)
    true_pos = K.cast(K.sum(good_preds * y_true), typecast)
    return true_pos


def _fp(y_true, y_pred, typecast='float32'):
    bad_preds = K.cast(tf.logical_not(K.equal(y_pred, y_true)), typecast)
    false_pos = K.cast(
        K.sum(bad_preds * K.cast(K.equal(y_true,0), typecast)), typecast)
    return false_pos


def _fn(y_true, y_pred, typecast='float32'):
    bad_preds = K.cast(tf.logical_not(K.equal(y_pred, y_true)), typecast)
    false_neg = K.cast(K.sum(bad_preds * y_true), typecast)
    return false_neg


def _tp_tn_fp_fn(y_true, y_pred):
    return _tp(y_true, y_pred), _tn(y_true, y_pred), \
        _fp(y_true, y_pred), _fn(y_true, y_pred)


def _fpr(fp, tn, eps=EPS):
    return fp / (fp + tn + eps)


def _fnr(fn, tp, eps=EPS):
    return fn / (fn + tp + eps)


def _tpr(fn, tp, eps=EPS):
    return 1 - _fnr(fn, tp, eps)


def _tnr(fp, tn, eps=EPS):
    return 1 - _fpr(fp, tn, eps)


def _fbeta(fp, fn, tp, beta2):
    return (1 + beta2) * tp / ((1 + beta2) * tp + beta2 * fn + fp + eps)


def _distance(fnr, fpr):
    return K.sqrt(K.square(fnr) + K.square(fpr))


def fpr(y_pred, y_true, threshold=0.5, eps=EPS):
    y_true, y_pred = _sanitize(y_true, y_pred, threshold=threshold)
    fp = _fp(y_pred, y_true)
    tn = _tn(y_pred, y_true)
    return _fpr(fp, tn)


def fnr(y_pred, y_true, threshold=0.5, eps=EPS):
    y_true, y_pred = _sanitize(y_true, y_pred, threshold=threshold)
    fn = _fn(y_pred, y_true)
    tp = _tp(y_pred, y_true)
    return _fnr(tn, tp, eps)


def tpr(y_pred, y_true, threshold=0.5, eps=EPS):
    return 1 - fnr(y_pred, y_true, threshold, eps)


def tnr(y_pred, y_true, threshold=0.5, eps=EPS):
    return 1 - fpr(y_pred, y_true, threshold, eps)


def fbeta(y_pred, y_true, beta=1, threshold=0.5, eps=EPS):
    y_true, y_pred = _sanitize(y_true, y_pred, threshold=threshold)
    fn = _fn(y_pred, y_true)
    tp = _tp(y_pred, y_true)
    fp = _fp(y_pred, y_true)
    beta2 = beta ** 2
    return _fbeta(fp, fn, tp, beta2, eps=eps)


def f1(y_pred, y_true,  threshold=0.5, eps=EPS):
    return fbeta(y_pred, y_true, beta=1, threshold=threshold, eps=eps)


def distance(y_pred, y_true, threshold=0.5, eps=EPS):
    y_true, y_pred = _sanitize(y_true, y_pred, threshold=threshold)
    tp, tn, fp, fn = _tp_tn_fp_fn(y_pred, y_true)
    fnr = _fnr(fn, tp, eps=eps)
    fpr = _fpr(fp, tn, eps=eps)
    return _distance(fnr, fpr)


def accuracy(y_true, y_pred, threshold=0.5, eps=EPS):
    y_true, y_pred = _sanitize(y_true, y_pred, threshold=threshold)
    return K.mean(K.equal(y_true, y_pred))


def compose(metrics, results, threshold=0.5, eps=EPS):
    """ Computes all specified metrics on results for each desired
    threshold value
    Args:
        metrics (list):
        results (tuple): (y_pred, y_true) either tf Tensor or ndarray
        threshold (int/tuple): list of values at each y_pred should be binarized
        eps: minimum value to avoid zero division error
    Returns:
        list of results at each given threshold
        Example: [((metric1, @thrs1), (metric1, @thrs2), (metric1, @thrs3)),
                  ((metric2, @thrs1), (metric2, @thrs2), (metric2, @thrs3))]
    """
    meter = []
    y_pred, y_true = results
    if not isinstance(threshold, (tuple, list)):
        threshold = (threshold,)

    n = len(threshold)
    for metric, thres in product(metrics, threshold):
        meter += [metric(y_pred, y_true, threshold=thres, eps=eps)]
    return list(zip(*[iter(meter)] * n))


class TruePos(Layer):
    """ Computes TP globally
    """

    def __init__(self, threshold=0.5, eps=EPS):
        super(TruePos, self).__init__(name='dis')
        self.stateful = True
        self.threshold = threshold
        self.tp = K.variable(0, dtype='float32')

    def reset_states(self):
        K.set_value(self.tp, 0)

    def __call__(self, y_true, y_pred):
        y_true, y_pred = _sanitize(y_true, y_pred, self.threshold)
        true_pos = _tp(y_true, y_pred)

        self.add_update(K.update_add(self.tp, true_pos),
                        inputs=[y_true, y_pred])

        return self.true_pos


class TrueNeg(Layer):
    """ Computes TN globally
    """

    def __init__(self, threshold=0.5, eps=EPS):
        super(TrueNeg, self).__init__(name='dis')
        self.stateful = True
        self.threshold = threshold
        self.tn = K.variable(0, dtype='float32')

    def reset_states(self):
        K.set_value(self.tn, 0)

    def __call__(self, y_true, y_pred):
        y_true, y_pred = _sanitize(y_true, y_pred, self.threshold)
        true_neg = _tn(y_true, y_pred)

        self.add_update(K.update_add(self.tn, true_neg),
                        inputs=[y_true, y_pred])

        return self.true_neg


class FalsePos(Layer):
    """ Computes FP globally
    """

    def __init__(self, threshold=0.5, eps=EPS):
        super(FalsePos, self).__init__(name='dis')
        self.stateful = True
        self.threshold = threshold
        self.fp = K.variable(0, dtype='float32')

    def reset_states(self):
        K.set_value(self.fp, 0)

    def __call__(self, y_true, y_pred):
        y_true, y_pred = _sanitize(y_true, y_pred, self.threshold)
        false_pos = _fp(y_true, y_pred)

        self.add_update(K.update_add(self.fp, false_pos),
                        inputs=[y_true, y_pred])

        return self.fp


class FalseNeg(Layer):
    """ Computes FN globally
    """

    def __init__(self, threshold=0.5, eps=EPS):
        super(Distance, self).__init__(name='dis')
        self.stateful = True
        self.threshold = threshold
        self.fn = K.variable(0, dtype='float32')

    def reset_states(self):
        K.set_value(self.fn, 0)

    def __call__(self, y_true, y_pred):
        y_true, y_pred = _sanitize(y_true, y_pred, self.threshold)
        false_neg = _fn(y_true, y_pred)

        self.add_update(K.update_add(self.fn, false_neg),
                        inputs=[y_true, y_pred])

        return self.fp


class FalsePosRate(Layer):
    """ Computes FPR globally
    """

    def __init__(self, threshold=0.5, eps=EPS):
        super(FalsePosRate, self).__init__(name='fpr')
        self.stateful = True
        self.threshold = threshold
        self.fp = K.variable(0, dtype='float32')
        self.tn = K.variable(0, dtype='float32')
        self.eps = eps

    def reset_states(self):
        K.set_value(self.fp, 0)
        K.set_value(self.tn, 0)

    def __call__(self, y_true, y_pred):
        y_true, y_pred = _sanitize(y_true, y_pred, self.threshold)
        false_pos = _fp(y_true, y_pred)
        true_neg = _tn(y_true, y_pred)

        self.add_update(K.update_add(self.fp, false_pos),
                        inputs=[y_true, y_pred])
        self.add_update(K.update_add(self.tn, true_neg),
                        inputs=[y_true, y_pred])

        return _fpr(self.fp, self.tn, eps=self.eps)


class FalseNegRate(Layer):
    """ Computes FNR globally
    """

    def __init__(self, threshold=0.5, eps=EPS):
        super(FalseNegRate, self).__init__(name='fnr')
        self.stateful = True
        self.threshold = threshold
        self.tp = K.variable(0, dtype='float32')
        self.fn = K.variable(0, dtype='float32')
        self.eps = eps

    def reset_states(self):
        K.set_value(self.tp, 0)
        K.set_value(self.fn, 0)

    def __call__(self, y_true, y_pred):
        y_true, y_pred = _sanitize(y_true, y_pred, self.threshold)
        true_pos = _tp(y_true, y_pred)
        false_neg = _fn(y_true, y_pred)

        self.add_update(K.update_add(self.tp, true_pos),
                        inputs=[y_true, y_pred])
        self.add_update(K.update_add(self.fn, false_neg),
                        inputs=[y_true, y_pred])

        return _fnr(self.fn, self.tp, eps=self.eps)


class FBetaScore(Layer):
    """ Computes F-beta score globally
    """

    def __init__(self, beta, threshold=0.5, eps=EPS):
        super(FBetaScore, self).__init__(name='f{:d}'.format(beta))
        self.stateful = True
        self.threshold = threshold
        self.tp = K.variable(0, dtype='float32')
        self.fn = K.variable(0, dtype='float32')
        self.fp = K.variable(0, dtype='float32')
        self.beta2 = beta ** 2
        self.eps = eps

    def reset_states(self):
        K.set_value(self.tp, 0)
        K.set_value(self.fn, 0)
        K.set_value(self.fp, 0)

    def __call__(self, y_true, y_pred):
        y_true, y_pred = _sanitize(y_true, y_pred, self.threshold)
        true_pos = _tp(y_true, y_pred)
        false_neg = _fn(y_true, y_pred)
        false_pos = _fp(y_true, y_pred)

        self.add_update(K.update_add(self.tp, true_pos),
                        inputs=[y_true, y_pred])
        self.add_update(K.update_add(self.fn, false_neg),
                        inputs=[y_true, y_pred])
        self.add_update(K.update_add(self.fp, false_pos),
                        inputs=[y_true, y_pred])

        return _fbeta(self.fp, self.fn, self.tp, self.beta2, eps=self.eps)


class Distance(Layer):
    """ Computes distance function globally
    """

    def __init__(self, threshold=0.5, eps=EPS):
        super(Distance, self).__init__(name='dis')
        self.stateful = True
        self.threshold = threshold
        self.tp = K.variable(0, dtype='float32')
        self.fp = K.variable(0, dtype='float32')
        self.tn = K.variable(0, dtype='float32')
        self.fn = K.variable(0, dtype='float32')
        self.eps = eps

    def reset_states(self):
        K.set_value(self.tp, 0)
        K.set_value(self.fn, 0)
        K.set_value(self.fp, 0)
        K.set_value(self.tn, 0)

    def __call__(self, y_true, y_pred):
        y_true, y_pred = _sanitize(y_true, y_pred, self.threshold)
        true_pos, true_neg, false_pos, false_neg = _tp_tn_fp_fn(y_true, y_pred)

        self.add_update(K.update_add(self.tp, true_pos),
                        inputs=[y_true, y_pred])
        self.add_update(K.update_add(self.tn, true_neg),
                        inputs=[y_true, y_pred])
        self.add_update(K.update_add(self.fn, false_neg),
                        inputs=[y_true, y_pred])
        self.add_update(K.update_add(self.fp, false_pos),
                        inputs=[y_true, y_pred])

        fpr = self.fp / (self.fp + self.tn + self.eps)
        fnr = self.fn / (self.fn + self.tp + self.eps)

        return K.sqrt(K.square(fnr) + K.square(fpr))


class ROC(object):
    """ Computes the Receiver Operating Characteristic (ROC) curve, and
        Area Under Curve for interpolated ROC, accumulating over the its calls.
        Returns the optimal threshold for the specified metric
    """
    def __init__(metric=distance, op='max'):
        self.inter_tprs = []
        self.tprs = []
        self.fprs = []
        self.aucs = []
        self.mean_tpr = None
        self.mean_auc = None
        self.std_tpr = None
        self.std_auc = None
        self.func = metric
        self.argcmp = np.argmax if op == 'max' else np.argmin
        self.mean_fpr = np.linspace(0, 1, 100)

    def __call__(self, y_true, proba):
        fpr, tpr, thresholds = roc_curve(y_true, proba)
        self.inter_tprs.append(interp(mean_fpr, fpr, tpr))
        self.tprs.append(tpr)
        self.fprs.append(fpr)

        self.inter_tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        self.aucs.append(roc_auc)
        dist = self.func(tpr, fpr)
        idx = self.argcmp(dist)

        return self.inter_tprs[-1][idx], dist[idx]

    def _mean():
        self.mean_tpr = np.mean(self.inter_tprs, axis=0)
        self.mean_auc = auc(self.mean_fpr, self.mean_tpr)
        self.mean_tpr[-1] = 1.0

    def _std():
        self.std_tpr = np.std(self.inter_tprs, axis=0)
        self.std_auc = np.std(self.aucs)

    def mean():
        if self.mean_tpr is None or self.mean_auc is None:
            self._mean()
        return self.mean_tpr, self.mean_auc

    def std():
        if self.std_tpr is None or self.std_auc is None:
            self._std()
        return self.std_tpr, self.std_auc

    def plot(filename='roc-crossval.eps', std=True):
        mean_tpr, mean_auc = self.mean()
        std_tpr, std_auc = self.std()

        plt.plot(self.mean_fpr, mean_tpr, color='b',
                label=r'ROC média (AUC = %0.2f $\pm$ %0.2f)' % \
                    (mean_auc, std_auc),
                lw=2, alpha=.8)

        if std == True:
            for i, (fpr, tpr) in enumerate(zip(self.fprs, self.tprs)):
                plt.plot(fpr, tpr, lw=1, alpha=0.3,
                    label='ROC fold %d (AUC = %0.2f)' % (i, self.aucs[i]))
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(self.mean_fpr, tprs_lower, tprs_upper,
                color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

        self.label_plot()
        if '.' not in filename:
            filename += '.eps'
        plt.savefig(filename, bbox_inches='tight')

    def label_plot():
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Taxa de Falso Positivo')
        plt.ylabel('Taxa de Verdadeiro Positivo')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
