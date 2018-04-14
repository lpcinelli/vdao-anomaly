import tensorflow as tf
from keras.layers import Layer
from keras import backend as K


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


class FalsePosRate(Layer):
    """ Computes FPR globally
    """

    def __init__(self, threshold=0.5, eps=1e-8):
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

        return self.fp / (self.fp + self.tn + self.eps)


class FalseNegRate(Layer):
    """ Computes FNR globally
    """

    def __init__(self, threshold=0.5, eps=1e-8):
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

        return self.fn / (self.fn + self.tp + self.eps)


class FBetaScore(Layer):
    """ Computes F-beta score globally
    """

    def __init__(self, beta, threshold=0.5, eps=1e-8):
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

        return (1 + self.beta2) * self.tp / \
            ((1 + self.beta2) * self.tp + self.beta2 * self.fn + self.fp + self.eps)


class Distance(Layer):
    """ Computes f1 score globally
    """

    def __init__(self, threshold=0.5, eps=1e-8):
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
