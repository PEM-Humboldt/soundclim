#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tune multiple classifiers

Created on 2020-01-15
Modified on 2020-01-27
@author: julloa@humboldt.org.co
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from classif_fcns import (tune_clf_rand, 
                          print_report, 
                          misclassif_idx, 
                          print_report_cv, 
                          pred_roi_to_file,
                          ismember)

## Set variables
path_df = '../data_training/traindb_shayii_features_mannot.txt'
n_splits_cv = 5  # for cross validation tuning
n_iter = 20  # nuber of iteration for randomized search

# load data
df = pd.read_csv(path_df)
df.dropna(axis=0, inplace=True)

df.lab_gt.value_counts()
df.lab_wname.value_counts()

# assign
X = df.loc[:,df.columns.str.startswith('shp')]
y_true = df.loc[:,'lab_gt']
clf_opt=['rf','svm','adb']

# train - tune clf
clf_tuned = dict()
for clf_name in clf_opt:
    clf_gs = tune_clf_rand(X, y_true, clf_name, 
                           n_splits_cv, n_iter=n_iter, 
                           score='f1_weighted')
    clf_tuned[clf_name] = clf_gs

## Test with cross validation
# Random Forest or Adaboost
clf = clf_tuned['rf'].best_estimator_
y_proba = cross_val_predict(clf, X, y_true, cv=n_splits_cv, 
                            method='predict_proba', verbose=True)
y_pred = y_proba[:,1]
print_report(y_true, y_pred, th=0.5, curve_type='roc')
 
# SVM
clf = clf_tuned['svm'].best_estimator_
y_pred = cross_val_predict(clf, X, y_true, cv=n_splits_cv, method='decision_function')
print_report(y_true, y_pred, th=0,curve_type='roc')


## Analyze the type of errors
df['pred'] = y_pred
th = 0.5
y_bin = np.array(y_pred>th,dtype=int)
idx = misclassif_idx(y_true, y_bin)
df.iloc[idx['fp']][['lab_wname','pred','obs']]
df.iloc[idx['fn']][['lab_wname','pred','obs']]


# Save tuned classifier, training data, filename of source code 
import sklearn
import joblib
fname_save_tuned_clf = '../data_clf/tuned_clf_20200131.joblib'
clf_tuned_persist = {'clf_tuned': clf_tuned,
                     'sklearn_version': sklearn.__version__,
                     'source_code': 'detector_hfaber/scripts/tune_clf_simple.py',
                     'training_data': {'X' : X, 'y_true' : y_true},
                     'description': 'Primeras pruebas de clasificacion basado en datasets en cortes. 07/01/2020'}

joblib.dump(clf_tuned_persist, fname_save_tuned_clf)
