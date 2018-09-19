import os
import cv2
import numpy as np
import torch



def evaluate_verification(feature_dir, metadata_dir):

    pairs = open(os.path.join(metadata_dir, "lfw/pairs.txt"))
    pairs.readline()

    # extract features
    left_features = []
    right_features = []    
    folds = []
    pair_types = []
    for i in range(10):
        for j in range(2):
            for k in range(300):
                line = pairs.readline()
                tokens = line.strip().split('\t')
                if j == 0:
                    left_path = os.path.join(feature_dir, tokens[0], '{:s}_{:0>4s}.txt'.format(tokens[0], tokens[1]))
                    right_path = os.path.join(feature_dir, tokens[0], '{:s}_{:0>4s}.txt'.format(tokens[0], tokens[2]))
                elif j == 1:
                    left_path = os.path.join(feature_dir, tokens[0], '{:s}_{:0>4s}.txt'.format(tokens[0], tokens[1]))
                    right_path = os.path.join(feature_dir, tokens[2], '{:s}_{:0>4s}.txt'.format(tokens[2], tokens[3]))
                left_feature = np.loadtxt(left_path)
                right_feature = np.loadtxt(right_path)
                folds.append(i)
                pair_types.append(j)
                left_features.append(left_feature)
                right_features.append(right_feature)
    left_features = np.array(left_features)
    right_features = np.array(right_features)
    folds = np.array(folds)
    pair_types = np.array(pair_types)

    fold_scores = []
    fold_thresholds = []
    for i in range(10):
        # split 10 folds into train & val set
        train_fold = (folds != i)
        val_fold = (folds == i)

        # subtract train mean and  normalize features
        mean = np.vstack([left_features[train_fold], right_features[train_fold]]).mean(axis=0)
        normalized_left_features = left_features - mean
        normalized_right_features = right_features - mean    
        normalized_left_features = normalized_left_features / np.linalg.norm(normalized_left_features, axis=1)[:, None]
        normalized_right_features = normalized_right_features / np.linalg.norm(normalized_right_features, axis=1)[:, None]


        pos_pairs = (pair_types == 0)
        neg_pairs = (pair_types == 1)

        # compute pairwise cosine distances for train and val set
        pairwise_distances = (normalized_left_features*normalized_right_features).sum(axis=1)

        # initialize corresponding thresholds for cosine distance
        train_thresholds = np.arange(-10000, 10001) / 10000
        train_accuracies = np.zeros_like(train_thresholds)

        # find best threshold on train set
        for key, value in enumerate(train_thresholds):
            num_train_true_pos = (pairwise_distances[np.logical_and(train_fold, pos_pairs)] > value).sum()
            num_train_true_neg = (pairwise_distances[np.logical_and(train_fold, neg_pairs)] < value).sum()
            train_accuracy = (num_train_true_pos + num_train_true_neg) / 5400 
            train_accuracies[key] = train_accuracy
        best_threshold = np.mean(train_thresholds[train_accuracies==train_accuracies.max()])

        # compute accuracy for test set
        num_val_true_pos = (pairwise_distances[np.logical_and(val_fold, pos_pairs)] > best_threshold).sum()
        num_val_true_neg = (pairwise_distances[np.logical_and(val_fold, neg_pairs)] < best_threshold).sum()
        fold_score = (num_val_true_pos + num_val_true_neg) / 600
        fold_thresholds.append(best_threshold)
        fold_scores.append(fold_score)
    fold_scores = np.array(fold_scores)
    fold_thresholds = np.array(fold_thresholds)

    return fold_thresholds, fold_scores
