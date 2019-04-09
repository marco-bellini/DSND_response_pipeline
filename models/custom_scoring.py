from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report

import sklearn.metrics as met
from sklearn.utils.fixes import signature
import pickle
import numpy as np



def make_scorers(weights,class_weights):
    scorerAP = make_scorer(mo_weighted_average_precision, greater_is_better = True,
                         class_weights=class_weights, adjust_for_frequency=True,
                         return_single=True)

    scorerCM = make_scorer(mo_weighted_cm_scorer, greater_is_better = True, weights=weights,
                         class_weights=class_weights, adjust_for_frequency=True,
                         return_single=True)

    scorers={'AP':scorerAP,'CM':scorerCM}
    print(scorerAP, scorerCM)
    return(scorers)




def mo_confusion_matrix(y_test, y_pred,as_DataFrame=False):
    """

    :param y_test:
    :param y_pred:
    :return:

    """
    cm = np.zeros((2, 2, y_test.shape[1])).astype(int)

    c = 0
    for column in y_test.columns:
        cm[:, :, c] = met.confusion_matrix(y_test[column], y_pred[:, c])
        c += 1
    tn = cm[0, 0, :].ravel()
    fp = cm[0, 1, :].ravel()
    fn = cm[1, 0, :].ravel()
    tp = cm[1, 1, :].ravel()

    if as_DataFrame:
        return( pd.DataFrame( np.vstack((tn, fp, fn, tp)).T,columns=['tn', 'fp', 'fn', 'tp'],index=y_test.columns ) )
    else:
        return (tn, fp, fn, tp)


def mo_weighted_average_precision(y_test, y_pred, class_weights=None,
                          adjust_for_frequency=False,
                          return_single=True):
    """

    :param y_test:
    :param y_pred:
    :param weights:
    :param class_weights:
    :param adjust_for_frequency:
    :param return_sum:
    :return:
    """
    # score with custom weights for errors

    score = met.average_precision_score(y_test, y_pred)

    if adjust_for_frequency:
        # adjusting the weights by the frequency

        real_disasters = y_test.sum()
        total_occurrences =  y_test.count()
        class_frequency= real_disasters / total_occurrences
        class_inv_frequencies = 1.0 / class_frequency
        class_inv_frequencies[real_disasters == 0] = 0

        # normalize the inv. frequencies
        class_inv_frequencies /= class_inv_frequencies.max()
        #         print()
        #         print(class_inv_frequencies)
        #         print()

        if not class_weights is None:
            class_weights *= class_inv_frequencies
        else:
            # the score is just adjusted by the inverse of frequency
            class_weights = class_inv_frequencies

    if not class_weights is None:
        # adjust score by weights
        score *= class_weights

    if return_single:
        #return np.sum(np.power(score.values,2.)))
        return np.sum(score.values)
    else:
        return score




def mo_weighted_cm_scorer(y_test, y_pred, weights={'tp': 1, 'tn': 1, 'fn': 1, 'fp': 1}, class_weights=None,
                          adjust_for_frequency=False,
                          return_single=True):
    """

    :param y_test:
    :param y_pred:
    :param weights:
    :param class_weights:
    :param adjust_for_frequency:
    :param return_sum:
    :return:
    """
    # score with custom weights for errors

    tn, fp, fn, tp = mo_confusion_matrix(y_test, y_pred)
    score = tn * weights['tn'] + tp * weights['tp'] - fn * weights['fn'] - fp * weights['tp']

    if adjust_for_frequency:
        # adjusting the weights by the frequency

        # real_disasters = tp + fn
        # class_inv_frequencies = 1.0 / real_disasters

        real_disasters = y_test.sum()
        total_occurrences =  y_test.count()
        class_frequency= real_disasters / total_occurrences
        class_inv_frequencies = 1.0 / class_frequency
        class_inv_frequencies[real_disasters == 0] = 0

        class_inv_frequencies[real_disasters == 0] = 0

        # normalize the inv. frequencies
        class_inv_frequencies /= class_inv_frequencies.max()
        #         print()
        #         print(class_inv_frequencies)
        #         print()

        if not class_weights is None:
            class_weights *= class_inv_frequencies
        else:
            # the score is just adjusted by the inverse of frequency
            class_weights = class_inv_frequencies

    if not class_weights is None:
        # adjust score by weights
        score *= class_weights

    if return_single:
        #return np.sum(np.power(score.values,2.)))
        return np.sum(score.values)
    else:
        return score

