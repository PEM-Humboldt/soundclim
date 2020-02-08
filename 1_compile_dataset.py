#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step by step script to compile a dataset to train and test a statistical 
classifier. Example for specific case:

    SPECIES: Scinax hayii
    SITE: BETANIA

Created on 2019-02-04
Latest modif on 2020-02-08
@author: julloa@humboldt.org.co
"""

import pandas as pd
import joblib
from maad.rois import find_rois_cwt
from maad import sound
from soundclim_utilities import (features_to_csv, 
                                 batch_find_rois, 
                                 batch_feature_rois, 
                                 listdir_pattern)

## 1. CALIBRATE FIND ROIS
# params Scinax hayii
flims = (1000, 4000); tlen = 0.3; th=1e-4
spec_opt = {'wl': 512, 'novlp': 256}
fname = '/Users/jsulloa/Dropbox/PostDoc/Soundclim/audio_sites/BETANIA/templates/BETA-_20161006_002000_section.wav'
# load and compute
s, fs = sound.load(fname)
_ = sound.spectrogram(s, fs, nperseg=spec_opt['wl'], db_range=250, display=True)
rois = find_rois_cwt(s, fs, flims, tlen, th, display=True, figsize=(13,6))
print(rois)


## 2. BATCH FIND ROIS
# Set variables
params_detection = {'flims' : (1000, 4000),
                     'tlen' : 0.30,
                     'th' : 1e-4}  # parametros S. hayii
path_audio = '/Users/jsulloa/Dropbox/PostDoc/Soundclim/audio_sites/BETANIA/train/'
path_save = './data_training/detections_shayii.joblib'
# Compute detections
flist = listdir_pattern(path_audio, ends_with='.wav')
flist = pd.DataFrame(flist, columns=['fname'])
detection_data = batch_find_rois(flist, params_detection, path_audio)
joblib.dump(detection_data, path_save)


## 3. BATCH FEATURE ROIS
# Set variables
params_features = {'flims' : (1000, 4000),
                   'opt_spec' : {'wl': 512, 'ovlp': 0.5, 'db_range': 250},
                   'opt_shape_str' : 'high'}
path_save = './data_training/features.joblib'
# Compute features
rois_list = detection_data['detections']
features = batch_feature_rois(rois_list, params_features, path_audio)
joblib.dump(features, path_save)
# Write to csv
features_to_csv(features, path_save[0:-7]+'.csv')


## 4. VISUALIZE
from sklearn import manifold
import matplotlib.pyplot as plt
df = pd.read_csv('./data_training/features.csv')
X = df.loc[:,df.columns.str.startswith('shp')]
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, 
                     n_iter=5000, verbose=1, perplexity=50)
Y = tsne.fit_transform(X)
plt.figure()
plt.scatter(Y[:,0], Y[:,1], alpha=0.2)


## 5. SELECT SAMPLES FOR ANNOTATION
# Filter rois by duration (0.2-0.5)
rois_tlen = df.max_t - df.min_t
idx_keep = (rois_tlen > 0.2) & (rois_tlen < 0.5)
df_sel = df.loc[idx_keep,:]

# cluster
from sklearn.cluster import KMeans
clf = KMeans(n_clusters=12, random_state=42)
X = df_sel.loc[:,df_sel.columns.str.startswith('shp')]
clf.fit(X)
df_sel.loc[:,'cluster'] = clf.labels_
# stratified sampling 
df_stratsample = df_sel.groupby('cluster', group_keys=False).apply(lambda x: x.sample(50))
df_stratsample.reset_index()
df_stratsample.to_csv('./data_training/df_stratsample.csv', index=False)