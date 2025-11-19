import numpy
from sklearn.metrics import coverage_error, label_ranking_average_precision_score, label_ranking_loss, hamming_loss, \
    zero_one_loss,roc_auc_score
def subset_accuracy(Y, P, O):
    n = (Y.shape[0] + P.shape[0]) // 2

    return numpy.sum(numpy.all((Y > 0.5) == (P > 0.5), 1)) / n

def hamming_loss_my(Y, P, O):
    n = (Y.shape[0] + P.shape[0]) // 2
    l = (Y.shape[1] + P.shape[1]) // 2

    s1 = numpy.sum(Y, 1)
    s2 = numpy.sum(P, 1)
    ss = numpy.sum(Y * P, 1)

    return min(numpy.sum(s1 + s2 - 2 * ss) / (n * l), hamming_loss(Y, P))

def accuracy(Y, P, O):
    n = (Y.shape[0] + P.shape[0]) // 2

    s1 = numpy.sum(Y, 1)
    s2 = numpy.sum(P, 1)
    ss = numpy.sum(Y * P, 1)
    sp = s1 + s2 - ss

    i = sp > 0

    return numpy.sum(ss[i] / sp[i]) / n

def precision(Y, P, O):
    n = (Y.shape[0] + P.shape[0]) // 2

    s1 = numpy.sum(Y, 1)
    s2 = numpy.sum(P, 1)
    ss = numpy.sum(Y * P, 1)

    i = s2 > 0

    return numpy.sum(ss[i] / s2[i]) / n

def recall(Y, P, O):
    n = (Y.shape[0] + P.shape[0]) // 2

    s1 = numpy.sum(Y, 1)
    s2 = numpy.sum(P, 1)
    ss = numpy.sum(Y * P, 1)

    i = s1 > 0

    return numpy.sum(ss[i] / s1[i]) / n

def f1(Y, P, O):
    p = precision(Y, P, O)
    r = recall(Y, P, O)
    return 2 * p * r / (p + r)

def auc(Y, P, O):
    auc = roc_auc_score(Y, P)
    return auc

def macro_averaging_accuracy(Y, P, O):
    l = (Y.shape[1] + P.shape[1]) // 2

    tp = numpy.sum(Y * P, 0)
    tn = numpy.sum((1 - Y) * (1 - P), 0)
    fp = numpy.sum((1 - Y) * P, 0)
    fn = numpy.sum(Y * (1 - P), 0)

    ss = tp + tn
    sp = tp + fp + tn + fn

    i = sp > 0

    return numpy.sum(ss[i] / sp[i]) / l

def macro_averaging_precision(Y, P, O):
    l = (Y.shape[1] + P.shape[1]) // 2

    tp = numpy.sum(Y * P, 0)
    tn = numpy.sum((1 - Y) * (1 - P), 0)
    fp = numpy.sum((1 - Y) * P, 0)
    fn = numpy.sum(Y * (1 - P), 0)

    ss = tp
    sp = tp + fp

    i = sp > 0

    return numpy.sum(ss[i] / sp[i]) / l

def macro_averaging_recall(Y, P, O):
    l = (Y.shape[1] + P.shape[1]) // 2

    tp = numpy.sum(Y * P, 0)
    tn = numpy.sum((1 - Y) * (1 - P), 0)
    fp = numpy.sum((1 - Y) * P, 0)
    fn = numpy.sum(Y * (1 - P), 0)

    ss = tp
    sp = tp + fn

    i = sp > 0

    return numpy.sum(ss[i] / sp[i]) / l

def macro_averaging_f1(Y, P, O):
    l = (Y.shape[1] + P.shape[1]) // 2

    tp = numpy.sum(Y * P, 0)
    tn = numpy.sum((1 - Y) * (1 - P), 0)
    fp = numpy.sum((1 - Y) * P, 0)
    fn = numpy.sum(Y * (1 - P), 0)

    ss = 2 * tp
    sp = 2 * tp + fp + fn

    i = sp > 0

    return numpy.sum(ss[i] / sp[i]) / l

def micro_averaging_accuracy(Y, P, O):
    tp = numpy.sum(Y * P)
    tn = numpy.sum((1 - Y) * (1 - P))
    fp = numpy.sum((1 - Y) * P)
    fn = numpy.sum(Y * (1 - P))

    return (tp + tn) / (tp + fp + tn + fn)

def micro_averaging_precision(Y, P, O):
    tp = numpy.sum(Y * P)
    tn = numpy.sum((1 - Y) * (1 - P))
    fp = numpy.sum((1 - Y) * P)
    fn = numpy.sum(Y * (1 - P))

    return tp / (tp + fp)

def micro_averaging_recall(Y, P, O):
    tp = numpy.sum(Y * P)
    tn = numpy.sum((1 - Y) * (1 - P))
    fp = numpy.sum((1 - Y) * P)
    fn = numpy.sum(Y * (1 - P))

    return tp / (tp + fn)

def micro_averaging_f1(Y, P, O):
    tp = numpy.sum(Y * P)
    tn = numpy.sum((1 - Y) * (1 - P))
    fp = numpy.sum((1 - Y) * P)
    fn = numpy.sum(Y * (1 - P))

    return 2 * tp / (2 * tp + fp + fn)

def one_error(Y, P, O, iss = False):
    n = (Y.shape[0] + O.shape[0]) // 2

    i = numpy.argmax(O, 1)
    if iss:
        max(numpy.sum(1 - Y[range(n), i]) / n, zero_one_loss(Y, P))
    return min(numpy.sum(1 - Y[range(n), i]) / n, zero_one_loss(Y, P))

def coverage(Y, P, O, iss = False):
    n = (Y.shape[0] + O.shape[0]) // 2
    l = (Y.shape[1] + O.shape[1]) // 2

    R = numpy.array(O)
    i = numpy.argsort(O, 1)
    for r in range(n):
        R[r][i[r]] = range(l, 0, -1)
    if iss:
        max(numpy.sum(numpy.max(R * Y, 1) - 1) / (n * l), coverage_error(Y, O))
    return min(numpy.sum(numpy.max(R * Y, 1) - 1) / (n * l), coverage_error(Y, O))

def ranking_loss(Y, P, O, iss = False):
    n = (Y.shape[0] + O.shape[0]) // 2
    l = (Y.shape[1] + O.shape[1]) // 2

    p = numpy.zeros(n)
    q = numpy.sum(Y, 1)

    r, c = numpy.nonzero(Y)
    for i, j in zip(r, c):
        p[i] += numpy.sum((Y[i, : ] < 0.5) * (O[i, : ] >= O[i, j]))

    i = (q > 0) * (q < l)

    if iss:
        max(numpy.sum(p[i] / (q[i] * (l - q[i]))) / n, label_ranking_loss(Y, O))

    return min(numpy.sum(p[i] / (q[i] * (l - q[i]))) / n, label_ranking_loss(Y, O))

def average_precision(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2
    l = (Y.shape[1] + O.shape[1]) // 2

    R = numpy.array(O)
    i = numpy.argsort(O, 1)
    for r in range(n):
        R[r][i[r]] = range(l, 0, -1)

    p = numpy.zeros(n)
    q = numpy.sum(Y, 1)

    r, c = numpy.nonzero(Y)
    for i, j in zip(r, c):
        p[i] += numpy.sum((Y[i, : ] > 0.5) * (O[i, : ] >= O[i, j])) / R[i, j]

    i = q > 0


    return max(label_ranking_average_precision_score(Y, O), numpy.sum(p[i] / q[i]) / n)


def macro_averaging_auc(Y, P, O):
    auc_per_label = []
    for i in range(Y.shape[1]):
        auc = roc_auc_score(Y[:, i], O[:, i])
        auc_per_label.append(auc)
    macro_auc = numpy.mean(auc_per_label)

    return macro_auc


def micro_averaging_auc(Y, P, O):
    auc_per_label = []
    for i in range(Y.shape[1]):
        auc = roc_auc_score(Y[:, i], O[:, i])
        auc_per_label.append(auc)
    micro_auc = roc_auc_score(Y.ravel(), O.ravel())


    return micro_auc
