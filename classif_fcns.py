#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 21:53:21 2018

@author: jsulloa
"""

import numpy as np
from scipy.stats import randint, uniform
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

def tune_clf_binary_grid(X, y_true, clf_name, n_splits_cv=5, refit_score='precision'):    
    """
    Tune a classifier using grid search cross validation
                
    """
    scorers = {
        'precision': metrics.make_scorer(metrics.precision_score),
        'recall': metrics.make_scorer(metrics.recall_score),
        'f1': metrics.make_scorer(metrics.f1_score),
        'auc': metrics.make_scorer(metrics.roc_auc_score)}
    
    if clf_name=='rf':
        print("Tuning Random Forest")
        clf = RandomForestClassifier(n_jobs=-1, class_weight='balanced_subsample')
        param_grid = {'n_estimators' : [1, 5, 10, 100, 300, 500],
                      'max_features' : [2, 6, 10, 14, 18, 32]}
    
    elif clf_name=='svm':
        print("Tuning Support Vector Machine")
        clf = svm.SVC(class_weight='balanced', probability=True)
        param_grid = [ {'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1], 'C': [0.1, 1, 10, 10]}]


    elif clf_name=='adb':
        print("Tuning Ada Boost")
        clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        param_grid = {'n_estimators':[50, 120], 
                      'learning_rate':[0.1, 0.5, 1.],
                      'base_estimator__min_samples_split' : np.arange(2, 8, 2),
                      'base_estimator__max_depth' : np.arange(1, 4, 1)}
    else:
        print("Invalid option. Valid options are: 'rf', 'adb' and 'svm' ")

    # Tune classifier with cross validation
    skf = StratifiedKFold(n_splits=n_splits_cv)
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, 
                               refit=refit_score, cv=skf, return_train_score=True,
                               iid=True, n_jobs=-1, verbose=2)

    # print basic info
    print('Best score:', grid_search.best_score_)
    print('Best parameters:', grid_search.best_params_)
    
    return grid_search

def tune_clf_multiclass_grid(X, y_true, clf_name, n_splits_cv=5,
                             score='f1_weighted', verbose=2, 
                             max_features_rf=[2, 6, 10, 14, 18]):    
    """
    Tune a classifier usinggrid search and cross validation
                
    """
    
    if clf_name=='rf':
        print("Tuning Random Forest")
        clf = RandomForestClassifier(n_jobs=-1)
        param_grid = {'n_estimators' : [1, 5, 10, 100, 300],
                      'max_features' : max_features_rf}
    
    elif clf_name=='svm':
        print("Tuning Support Vector Machine")
        clf = svm.SVC()
        param_grid = {'kernel': ['rbf'], 
                      'gamma': [0.001, 0.01, 0.1, 1], 
                      'C': [0.1, 1, 10, 100]}

    elif clf_name=='adb':
        print("Tuning Ada Boost")
        clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        param_grid = {'n_estimators':[50, 120], 
                      'learning_rate':[0.1, 0.5, 1.],
                      'base_estimator__min_samples_split' : np.arange(2, 8, 2),
                      'base_estimator__max_depth' : np.arange(1, 4, 1)}
    else:
        print("Invalid option. Valid options are: 'rf', 'svm' and 'adb'")

    # Tune classifier with cross validation
    skf = StratifiedKFold(n_splits=n_splits_cv)
    grid_search = GridSearchCV(clf, param_grid, scoring=score, 
                               refit=True, cv=skf, return_train_score=True, 
                               iid=True, n_jobs=-1, verbose=2)

    # print basic info
    print('Best score:', grid_search.best_score_)
    print('Best parameters:', grid_search.best_params_)
    return grid_search

## NOTE_ NAME WAS CHANGED FROM tune_clf_multiclass_rand
def tune_clf_rand(X, y_true, clf_name, n_splits_cv=5,
                  n_iter=10, score='f1_weighted', verbose=2):    
    """
    Tune a classifier using randomized search and cross validation
    
    Parameters: 
        X: array-like, dtype=float64, size=[n_samples, n_features]
           array with observations and features
        y_true: array, dtype=float64, size=[n_samples]
            array with labels for each observation
        clf_name: str
            name of the classifier to be tuned, 'rf', 'adb' or 'svm'.
        n_splits_cv: int
            Number of folds for cross validation
        n_iter: int, default=10
            Number of parameter settings that are sampled
        score: string, callable, list/tuple, dict or None, default: None
            Score to evaluate prediction on the test set
        verbose: int
            Controls the verbosity: the higher, the more messages
    """
    
    if clf_name=='rf':
        print("Tuning Random Forest")
        clf = RandomForestClassifier(n_jobs=-1)
        param_grid = {'max_depth' : [3, None], 
                      'n_estimators' : randint(1,1000),
                      'max_features' : randint(1,X.shape[1]-1)}
    
    elif clf_name=='svm':
        print("Tuning Support Vector Machine")
        clf = svm.SVC(class_weight='balanced', probability=True)
        param_grid = {'kernel': ['rbf'], 
                      'gamma': uniform(0.01, 1), 
                      'C': uniform(1,100)}

    elif clf_name=='adb':
        print("Tuning Ada Boost")
        clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        param_grid = {'n_estimators':randint(1,1000), 
                      'learning_rate':uniform(0.01, 1),
                      'base_estimator__min_samples_split' : randint(2,20),
                      'base_estimator__max_depth' : [3, None]}

    elif clf_name=='knn':
        print('Tuning KNN')
        clf = KNeighborsClassifier()
        param_grid = { 'n_neighbors': randint(1,50)}

    elif clf_name=='cnb':
        print('Tuning Complement Naive Bayes')
        clf = ComplementNB()
        param_grid = { 'alpha': uniform(0, 10)}
    
    elif clf_name=='gnb':
        print('Tuning Gaussian Naive Bayes')
        clf = GaussianNB()
        param_grid = { 'var_smoothing': uniform(1e-10, 10)}
    
    elif clf_name=='ann':
        print('Tuning Artificial Neural Networks')
        clf = MLPClassifier(solver='lbfgs', max_iter=500)
        param_grid = {'hidden_layer_sizes' :randint(5,100),
                      'alpha': uniform(1e-5,1)}

    else:
        print("Invalid option. Valid options are: 'rf', 'svm' and 'adb'")

    # Tune classifier with cross validation
    skf = StratifiedKFold(n_splits=n_splits_cv)
    rand_search = RandomizedSearchCV(clf, param_grid, scoring=score, n_iter=n_iter,
                                     refit=True, cv=skf, return_train_score=True, 
                                     iid=True, n_jobs=-1, verbose=2)
    rand_search.fit(X, y_true)
    # print basic info
    print('Best score:', rand_search.best_score_)
    print('Best parameters:', rand_search.best_params_)
    
    return rand_search


def print_report_cv(clf_gs):
    """
    Print report of GridSearch
    Accepts only numerical y_true
                
    """
    print("Grid scores on development set:")
    print()
    df_scores = pd.DataFrame.from_dict(clf_gs.cv_results_)
    print(df_scores[['mean_fit_time',
               'mean_test_f1', 
               'mean_test_recall',
               'mean_test_precision',
               'param_max_features',
               'param_n_estimators']])
    print()
    print('Best parameters:')
    print("\n".join("{}\t{}".format(k, v) for k, v in clf_gs.best_params_.items()))
    print('Best score:', np.round(clf_gs.best_score_,3))
    

def plot_param_cv(clf_gs, scorer, param):
    df_res = pd.DataFrame.from_dict(clf_gs.cv_results_) 
    score = 'mean_test_' + scorer
    score_std = 'std_test_' + scorer
    param = 'param_' + param
    mean_value = df_res[score]
    std_value = df_res[score_std]
    # plot 
    plt.figure(figsize=(8, 6))
    plt.errorbar(np.arange(len(mean_value)), mean_value, yerr=std_value, fmt='o')
    plt.xticks(np.arange(len(mean_value)), df_res[param])
    plt.xlabel(param)
    plt.ylabel(score)
    plt.box(on=None)
    

def print_report(y_true, y_pred, th=0.5, plot=True, curve_type='roc'):
    """
    Print a report of binary classification performance
    
    Parameters
    ----------
        y_true: ndarray
            Ground truth data
        y_pred: ndarray
            Predicted data
        th: float
            Thrshold to compute metrics precision, recall, accuracy and f1 score.
        plot: bool, default True
            Plot curves
        curve_type: string, default 'precision_recall'
            Type of curve to plot, 'precision_recall' or 'roc'
    Returns
    -------
        model_eval: dict
            Dictionary with multiple metrics for model evaluation
            
    Note from Hands on Machine Learning with scikit-learn ():
        Since the ROC curve is so similar to the precision/recall (or PR) 
        curve, you may wonder how to decide which one to use. As a rule of 
        thumb, you should prefer the PR curve whenever the positive class is 
        rare or when you care more about the false positives than the false 
        negatives, and the ROC curve otherwise.
    """
    y_bin = y_pred>th
    fpr, tpr, th_roc = metrics.roc_curve(y_true, y_pred, pos_label=1)
    precisions, recalls, th_pr = metrics.precision_recall_curve(y_true, y_pred, 
                                                       pos_label=1)
    
    model_eval = {'auc' : metrics.auc(fpr, tpr),
                  'th' : th,
                  'confusion_matrix' : metrics.confusion_matrix(y_true, y_bin),
                  'precision': metrics.precision_score(y_true, y_bin), 
                  'recall': metrics.recall_score(y_true, y_bin),
                  'accuracy': metrics.accuracy_score(y_true, y_bin),
                  'f1': metrics.f1_score(y_true, y_bin)}
    
    print()
    print('Area Under the Curve:',np.round(metrics.auc(fpr, tpr),decimals=4))
    print('\nConfusion matrix for threshold =', th,':')
    print(pd.DataFrame(metrics.confusion_matrix(y_true, y_bin),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    print()
    print(pd.DataFrame({
        'Precision': [metrics.precision_score(y_true, y_bin)], 
        'Recall': [metrics.recall_score(y_true, y_bin)],
        'Accuracy': [metrics.accuracy_score(y_true, y_bin)],
        'F1 score': [metrics.f1_score(y_true, y_bin)]}))
    
    # Trace ROC curve
    if plot==True and curve_type=='roc':
        plt.figure(figsize= (16,8))
        plt.subplot(1,2,1)  # left size
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.title('ROC curve')
        plt.xlabel('False postivie rate')
        plt.ylabel('True positive rate')
        plt.grid(1)        
        plt.subplot(1,2,2)  # right size
        plt.plot(th_roc, 1-fpr, label='1-False positive rate')
        plt.plot(th_roc, tpr, label='True positive rate')
        plt.xlim(-0.5,1.1)
        plt.xlabel('Decision threshold')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.show()

    elif plot==True and curve_type=='prc':
        plt.figure(figsize= (16,8))
        plt.subplot(1,2,1)  # left size
        plt.plot([1, 0], [0, 1], 'k--')
        plt.plot(recalls, precisions)
        plt.title('PR curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(1)        
        plt.subplot(1,2,2)  # right size
        plt.plot(th_pr, precisions[:-1], label='Precision')
        plt.plot(th_pr, recalls[:-1], label='Recall')
        plt.xlim(-0.1,1.1)
        plt.xlabel('Decision threshold')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.show()
    else:
        print('Error plotting: curve_type should be \'roc\' or \'prc\' ')
        pass
    
    return model_eval

def print_report_grid_search(clf, clf_name, X_test, y_test):
    """
    Print report of classifier performance evaluated on new test data
    
    Parameters
    ----------
        clf : classifier previously tuned with GridSearchCV
        clf_name: name of classifier. Valid options are 'svm', 'rf' ,'adb'
        X_test: features for test data
        y_test: labels for test data
    Returns
    -------
        clf_gs: a tune classifier with GridSearchCV.
            
    """
    # make the predictions
    y_pred = clf.predict(X_test.values)
    ## Print report
    print('Best params:')
    print(clf.best_params_)
    
    # confusion matrix on the test data.
    print('\nConfusion matrix optimized on test data:')
    print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    print('\nPrecision:', metrics.precision_score(y_test, y_pred), 
          '\nRecall:', metrics.recall_score(y_test, y_pred),
          '\nAccuracy:', metrics.accuracy_score(y_test, y_pred))

    # Compute performance at multiple thresholds
    if clf_name=='rf':
        y_test_score = clf.predict_proba(X_test)
        fpr, tpr, th = metrics.roc_curve(y_test, y_test_score[:,1], pos_label=1)
    elif clf_name=='svm':
        y_test_score = clf.decision_function(X_test)
        fpr, tpr, th = metrics.roc_curve(y_test, y_test_score, pos_label=1)
    elif clf_name=='adb':
        y_test_score = clf.decision_function(X_test)
        fpr, tpr, th = metrics.roc_curve(y_test, y_test_score, pos_label=1)
    # Trace ROC curve
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=clf_name)
    plt.grid(1)

def misclassif_idx(y_true, y_pred):
    """
    Get indices of misclassified observations
    
    Parameters
    ----------
        y_true: ndarray, numeric
            Ground truth labels in numeric format
        y_pred: ndarray, numeric
            Predicted labels as numeric, 0 or 1
    Returns
    -------
    """
    idx_fp=np.where(0 > y_true - y_pred)
    idx_fn=np.where(0 < y_true - y_pred)
    return {'fp':idx_fp[0], 'fn': idx_fn[0]}

from sklearn.metrics import f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def test_feature_set(fname_db):
    """ Test feature set discrimination with Linear Discrimanant Analysis
    
    TODO: Check if the approach is valid, set plot functionality plot=1
    """
    # load file and assign to objects
    df = pd.read_csv(fname_db)
    df = df.dropna()
    df = df.reset_index(drop=True)
    shape_idx = [col for col in df if col.startswith('shp')]
    X = df[shape_idx]
    y_true = df.label.str.slice(0,1)
    y_true = y_true.astype('int8')

    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    y_pred = lda.fit(X, y_true).predict(X)
    score = round(f1_score(y_pred=y_pred,y_true=y_true),3)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit(X, y_true).transform(X)
    plt.figure()
    plt.plot(X_lda[y_true==0],'o',color='navy', markersize=3)
    plt.plot(X_lda[y_true==1],'o',color='darkorange', markersize=3)
    plt.title('LDA on: '+fname_db)
    plt.grid(True)
    plt.show()
    print('F1 score: ', score)

def get_max_df(df, column):
    """
    Get maximum of a dataframe column
    
    Parameters:
    ----------
        df: pandas dataframe
            A data frame with multiple columns
        column: str
            Name of the column to get the maximum
    Returns:
    -------
    """
    idx_max = df[column].idxmax()
    return df.loc[idx_max,:]

def pred_roi_to_file(path_pred, column_proba):
    """
    Translate predictions on ROIs to file by keeping the ROI with maximum
    probability. Note: This works only for binary classifiers
    
    Parameters:
    ----------
        path_pred: str
            Full path to the ROIs predicted (xdata)
        column_proba: str
            Name of the column that has the positive predictions
    Returns:
    -------
        y_pred_file: dataframe
            Dataframe with name of files and the associated positive prediction

    """
    # load file and assign to objects
    y_pred_roi = pd.read_csv(path_pred)
    
    # group by fname
    aux = y_pred_roi.groupby('fname')    
    splited_y_pred = [aux.get_group(x) for x in aux.groups]
    
    # get maximum score for each file and make a dataframe
    y_pred_file = list(map(lambda x: get_max_df(x, column_proba), splited_y_pred))
    y_pred_file = pd.concat(y_pred_file, axis=1)
    y_pred_file = y_pred_file.transpose()
    y_pred_file = y_pred_file.sort_values('fname').reset_index(drop=True)
    return y_pred_file

def ismember(a, b, bool_array=False):
    """
    Get indices of elements of a that are in b

    Parameters:
    ----------
        a: list or pandas series
            list with elements to search in b
        b: list or pandas series
            target list
        bool_array: bool
            Select type of output, boolean or index based.
    Returns:
    -------
        idx: ndarray
            Numpy array containing indices where the data in A is found in B,
            and None where the element is not present.
            If bool_array is selected, logical True where the data is present,
            and False where is absent.
    """
    bind = {}
    
    if bool_array==True:  # return bool array
        for i, elt in enumerate(b):
            if elt not in bind:
                bind[elt] = True
                array_out = [bind.get(itm, False) for itm in a]
    
    else:  # return index array
        for i, elt in enumerate(b):
            if elt not in bind:
                bind[elt] = i
                array_out = [bind.get(itm, None) for itm in a]
    
    return np.array(array_out)



def df_max_proba(df_in, cols):
    """
    Summarizes a dataframe by using the maximum value of each column. Specially
    designed for y_pred dataframes that have a 'fname' column associated with
    labels and probabilities
    
    Parameters
    -----------
        df_in : Dataframe
            The DataFrame has to have a 'fname' column and associated probabilities
    Returns
    -------
        df_out : Dataframe
            A single row with 'fname' and maximum probabilities for each 
            class label
        
    """
    aux_df = df_in[cols]
    df_out = pd.Series({'fname': df_in.fname.unique()[0]})
    df_out = df_out.append(aux_df.max().to_frame())
    return df_out.T

def plot_roc_curve(fpr,tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
def print_report_multiclass(y_true, y_pred):
    labels = y_true.unique()
    print('\n')
    print('Confusion matrix:\nOracle labels as rows and predicted lables as columns\n')
    print(pd.DataFrame(metrics.confusion_matrix(y_true, y_pred, labels),
                       columns=labels, index=labels))
    print('\n')
    print(pd.DataFrame({
        'Precision': [metrics.precision_score(y_true, y_pred, average='weighted')], 
        'Recall': [metrics.recall_score(y_true, y_pred, average='weighted')],
        'Accuracy': [metrics.accuracy_score(y_true, y_pred)],
        'F1 score': [metrics.f1_score(y_true, y_pred, average='macro')]}))

def feature_importance(clf, feature_names):
    """
    Give sorted features according to their importance in the classification

    Parameters
    ----------
        clf : sklearn estimator
            fitted classifier with attribute "feature_importances_"
        feature_names: str
            name of the features used for classification. Can be obtained as
            X.columns if X is the features dataframe.
    Returns
    -------
        feature_importance: list
            Sorted list with importances in decreasing order
    """
    f_imp = sorted(zip(clf.feature_importances_, feature_names), reverse=True)
    return f_imp