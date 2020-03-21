#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch predict presence absence of sounds of interest

VERSION 0.2
Created on April 2019 (version 0.1)
Modified on May 2019 (version 0.2)
@author: jseb.ulloa@gmail.com
"""
## Import modules
import pandas as pd
import joblib
from classif_fcns import tune_clf_rand, print_report, misclassif_idx
from soundclim_utilities import batch_predict_rois, listdir_pattern, predictions_to_df
import settings

# ----------  Set variables -------- #
path_audio_db =  '/Users/jsulloa/Dropbox/PostDoc/Soundclim/audio_sites/BETANIA/test/'  # Location of audio files
path_tuned_clf='../models/tuned_clf_20200131.joblib' # location of classifiers
#path_save_predictions =  '../scinhayii_alt/predfile_campobelo.pkl' # filename where the predictions are to be saved
# ----------------------------------- #

# load filelist and set format
flist = listdir_pattern(path_audio_db, ends_with='.wav')
flist = pd.DataFrame(flist, columns=['fname'])
#flist = flist.loc[0:9]  # for testing purposes uncomment this line


# Fixed variables, DO NOT modify
params = {'flims' : settings.detection['flims'],
          'tlen' : settings.detection['tlen'],
          'th' : settings.detection['th'],
          'opt_spec' : settings.features['opt_spec'],
          'sample_rate_wav': settings.features['sample_rate'],
          'opt_shape_str' : settings.features['opt_shape_str']}

# Batch predict on files
clfs_data = joblib.load(path_tuned_clf)
tuned_clfs = clfs_data['clf_tuned']
predictions = batch_predict_rois(flist, tuned_clfs, params, path_audio_db)

# Evaluate presence-absence predictions
gt = pd.read_csv('../data_test/mannot_files_scinhayii_betania.csv')
y_pred = predictions_to_df(predictions,  ['1.0_svm', '1.0_rf', '1.0_adb'])

# sort elements according to filename
y_true_file = gt.sort_values('fname').reset_index(drop=True)
y_pred_file = y_pred.sort_values('fname').reset_index(drop=True)

# Check manually
if y_true_file.fname.equals(y_pred_file.fname):
    print('All elements are equal')
else:
    print('Elements are not equal, check manually')
    print('pd.concat({\'true\':y_true_file.fname,\'pred\':y_pred_file.fname}, axis=1')

print_report(y_true_file['lab_gt'], y_pred_file['1.0_svm'], th=0.5, curve_type='roc')