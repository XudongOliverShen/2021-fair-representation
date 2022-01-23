"""The code was copied from liorshk's 'face_pytorch' repository:
    https://github.com/liorshk/facenet_pytorch/blob/master/eval_metrics.py

    Which in turn was copied from David Sandberg's 'facenet' repository:
        https://github.com/davidsandberg/facenet/blob/master/src/lfw.py#L34
        https://github.com/davidsandberg/facenet/blob/master/src/facenet.py#L424

    Modified to also compute precision and recall metrics.
"""

import numpy as np
from sklearn.metrics import auc
from sklearn.model_selection import KFold
from scipy import interpolate
from tqdm import tqdm


def evaluate_lfw(distances, labels, num_folds=10, far_target=1e-3):
    """Evaluates on the Labeled Faces in the Wild dataset using KFold cross validation based on the Euclidean
    distance as a metric.

    Note: "TAR@FAR=0.001" means the rate that faces are successfully accepted (True Accept Rate) (TP/(TP+FN)) when the
    rate that faces are incorrectly accepted (False Accept Rate) (FP/(TN+FP)) is 0.001 (The less the FAR value
    the mode difficult it is for the model). i.e: 'What is the True Positive Rate of the model when only one false image
    in 1000 images is allowed?'.
        https://github.com/davidsandberg/facenet/issues/288#issuecomment-305961018

    Args:
        distances: numpy array of the pairwise distances calculated from the LFW pairs.
        labels: numpy array containing the correct result of the LFW pairs belonging to the same identity or not.
        num_folds (int): Number of folds for KFold cross-validation, defaults to 10 folds.
        far_target (float): The False Accept Rate to calculate the True Accept Rate (TAR) at,
                             defaults to 1e-3.
    Returns:
        true_positive_rate: Mean value of all true positive rates across all cross validation folds for plotting
                             the Receiver operating characteristic (ROC) curve.
        false_positive_rate: Mean value of all false positive rates across all cross validation folds for plotting
                              the Receiver operating characteristic (ROC) curve.
        accuracy: Array of accuracy values per each fold in cross validation set.
        precision: Array of precision values per each fold in cross validation set.
        recall: Array of recall values per each fold in cross validation set.
        roc_auc: Area Under the Receiver operating characteristic (ROC) metric.
        best_distances: Array of Euclidean distance values that had the best performing accuracy on the LFW dataset
                         per each fold in cross validation set.
        tar: Array that contains True Accept Rate values per each fold in cross validation set
              when far (False Accept Rate) is set to a specific value.
        far: Array that contains False accept rate values per each fold in cross validation set.
    """

    # # Calculate ROC metrics
    # thresholds_roc = np.arange(min(distances)-0.01, max(distances)+0.01, 0.002)
    # true_positive_rate, false_positive_rate, precision, recall, accuracy, best_distances = \
    #     calculate_roc_values(
    #         thresholds=thresholds_roc, distances=distances, labels=labels, num_folds=num_folds
    #     )

    # roc_auc = auc(false_positive_rate, true_positive_rate)

    # Calculate validation rate
    dis_max = distances.max()
    dis_min = distances.min()
    thresholds_val = np.arange(dis_min, dis_max, 0.1)
    tar, far = calculate_val(
        thresholds_val=thresholds_val, distances=distances, labels=labels, far_target=far_target, num_folds=num_folds,
        shuffle=True
    )

    # return true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances,\
    #     tar, far
    return tar.mean(), far.mean()


def calculate_roc_values(thresholds, distances, labels, num_folds=10):
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    true_positive_rates = np.zeros((num_folds, num_thresholds))
    false_positive_rates = np.zeros((num_folds, num_thresholds))
    precision = np.zeros(num_folds)
    recall = np.zeros(num_folds)
    accuracy = np.zeros(num_folds)
    best_distances = np.zeros(num_folds)

    indices = np.arange(num_pairs)

    for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best distance threshold for the k-fold cross validation using the train set
        accuracies_trainset = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds):
            _, _, _, _, accuracies_trainset[threshold_index] = calculate_metrics(
                threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
            )
        best_threshold_index = np.argmax(accuracies_trainset)

        # Test on test set using the best distance threshold
        for threshold_index, threshold in enumerate(thresholds):
            true_positive_rates[fold_index, threshold_index], false_positive_rates[fold_index, threshold_index], _, _,\
                _ = calculate_metrics(
                    threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set]
                )

        _, _, precision[fold_index], recall[fold_index], accuracy[fold_index] = calculate_metrics(
            threshold=thresholds[best_threshold_index], dist=distances[test_set], actual_issame=labels[test_set]
        )

        best_distances[fold_index] = thresholds[best_threshold_index]
    true_positive_rate = np.mean(true_positive_rates, 0)
    false_positive_rate = np.mean(false_positive_rates, 0)

    return true_positive_rate, false_positive_rate, precision, recall, accuracy, best_distances


def calculate_metrics(threshold, dist, actual_issame):
    # If distance (similarity) is greater than threshold, then prediction is set to True
    predict_issame = np.less(threshold, dist)

    true_positives = np.sum(np.logical_and(predict_issame, actual_issame))
    false_positives = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    true_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # For dealing with Divide By Zero exception
    true_positive_rate = 0 if (true_positives + false_negatives == 0) else \
        float(true_positives) / float(true_positives + false_negatives)

    false_positive_rate = 0 if (false_positives + true_negatives == 0) else \
        float(false_positives) / float(false_positives + true_negatives)

    precision = 0 if (true_positives + false_positives) == 0 else\
        float(true_positives) / float(true_positives + false_positives)

    recall = 0 if (true_positives + false_negatives) == 0 else \
        float(true_positives) / float(true_positives + false_negatives)

    accuracy = float(true_positives + true_negatives) / dist.size

    return true_positive_rate, false_positive_rate, precision, recall, accuracy


def calculate_val(thresholds_val, distances, labels, far_target=1e-3, num_folds=10, shuffle=False):
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds_val)
    k_fold = KFold(n_splits=num_folds, shuffle=shuffle)

    tar = np.zeros(num_folds)
    far = np.zeros(num_folds)

    indices = np.arange(num_pairs)
    for fold_index, (train_set, test_set) in tqdm(enumerate(k_fold.split(indices))):
        # Find the euclidean distance threshold that gives false acceptance rate (far) = far_target
        # far_train = np.zeros(num_thresholds)
        # for threshold_index, threshold in enumerate(thresholds_val):
        #     _, far_train[threshold_index] = calculate_val_far(
        #         threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
        #     )
        # if np.max(far_train) >= far_target:
        #     f = interpolate.interp1d(far_train, thresholds_val, kind='slinear')
        #     threshold = f(far_target)
        # else:
        #     threshold = 0.0
        
        thresh_1 = -1
        thresh_2 = 1
        far_1 = 1
        far_2 = 0

        while thresh_2 - thresh_1 > 0.001:
            new_thresh = (thresh_2 + thresh_1) / 2
            _, new_far = calculate_val_far(
                threshold=new_thresh, dist=distances[train_set], actual_issame=labels[train_set]
            )
            if new_far > far_target:
                thresh_1 = new_thresh
                far_1 = new_far
            else:
                thresh_2 = new_thresh
                far_2 = new_far

        tar[fold_index], far[fold_index] = calculate_val_far(
            threshold=new_thresh, dist=distances[test_set], actual_issame=labels[test_set]
        )

    return tar, far


def calculate_val_far(threshold, dist, actual_issame):
    # If distance (similarity) is less than threshold, then prediction is set to True
    predict_issame = np.less(threshold, dist)

    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    num_same = np.sum(actual_issame)
    num_diff = np.sum(np.logical_not(actual_issame))

    if num_diff == 0:
        num_diff = 1
    if num_same == 0:
        return 0, 0

    tar = float(true_accept) / float(num_same)
    far = float(false_accept) / float(num_diff)

    return tar, far