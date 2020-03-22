#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BATCH PREDICT 
Presence absence of sounds of soundmarks

VERSION 0.3
Created on April 2019 (version 0.1)
Modified on May 2019 (version 0.2)
@author: jseb.ulloa@gmail.com
"""

import pandas as pd
import joblib
import settings
from classif_fcns import print_report, misclassif_idx
from soundclim_utilities import (features_to_csv, 
                                 batch_find_rois, 
                                 batch_feature_rois, 
                                 listdir_pattern)


## 1. BATCH FIND ROIS
flist = listdir_pattern(settings.path_audio['test'], ends_with='.wav')
flist = pd.DataFrame(flist, columns=['fname'])
detection_data = batch_find_rois(flist, settings.detection, 
                                 settings.path_audio['test'])

## 2. BATCH FEATURE ROIS
rois_list = detection_data['detections']
features = batch_feature_rois(rois_list, settings.features, settings.path_audio['test'])
df = features_to_csv(features)
# Filter rois by duration
tlen_min, tlen_max = settings.selrois['tlen_lims']
rois_tlen = df.max_t - df.min_t
idx_keep = (rois_tlen > tlen_min) & (rois_tlen < tlen_max)
df = df.loc[idx_keep,:]
# Write features dataframe to disk
df.to_csv('../models/test_rois_features.csv', index=False, sep=',')

## 3. PREDICT ROIS
tuned_clfs = joblib.load('../models/tuned_clf_20200316.joblib')
clf = tuned_clfs['clf_tuned']['rf']
X = df.loc[:,df.columns.str.startswith('shp')]
df['proba'] = clf.predict_proba(X)[:,1]


## 4. ARRANGE PREDICTIONS ROIS -> FILE
# arrange output by file
pred_file = df.loc[:,['fname','proba']].groupby('fname').max()
pred_file.reset_index(inplace=True)
# append files with no detections
files_norois = flist.loc[~flist.fname.isin(pred_file.fname),:]
files_norois['proba'] = -1
pred_file = pred_file.append(files_norois, ignore_index=True)
pred_file.sort_values(by='fname', inplace=True)
# load ground truth file
gt_file = pd.read_csv('../data/test/mannot_files_scinhayii_betania.csv')
# sort elements according to filename
gt_file = gt_file.sort_values('fname').reset_index(drop=True)
pred_file = pred_file.sort_values('fname').reset_index(drop=True)
# check correct alignment
if gt_file.fname.equals(pred_file.fname):
    print('All elements are equal')
    pred_file['lab_gt'] = gt_file.lab_gt
else:
    print('Elements are not equal, check manually')
    print('pd.concat({\'true\':y_true_file.fname,\'pred\':y_pred_file.fname}, axis=1')


## 5. CHECK PERFORMANCE AND TYPES OF ERROR
th = 0.5
print_report(pred_file['lab_gt'], pred_file['proba'], th=th, curve_type='roc')

## Analyze the type of errors
y_bin = np.array(pred_file.proba>th, dtype=int)
idx = misclassif_idx(gt_file.lab_gt, y_bin)
pred_file.iloc[idx['fp']][['fname','proba', 'lab_gt']]
pred_file.iloc[idx['fn']][['fname','proba', 'lab_gt']]
