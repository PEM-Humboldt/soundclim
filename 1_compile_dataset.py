#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step by step script to compile a dataset to train and test a statistical 
classifier. Example for specific case:
Created on 2019-02-04
Latest modif on 2020-02-09
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

## LOAD SETTINGS
# All settings are stored in the module settings_compile_dataset.py
from settings import load_compile_dataset
params_detection, params_features, params_selrois = load_compile_dataset()


## 1. CALIBRATE FIND ROIS
s, fs = sound.load(params_detection['path_template'])
rois = find_rois_cwt(s, fs, 
                     flims = params_detection['flims'], 
                     tlen = params_detection['tlen'], 
                     th = params_detection['th'], 
                     display=True, figsize=(13,6))
print(rois)


## 2. BATCH FIND ROIS
flist = listdir_pattern(params_detection['path_audio'], ends_with='.wav')
flist = pd.DataFrame(flist, columns=['fname'])
detection_data = batch_find_rois(flist, params_detection, 
                                 params_detection['path_audio'])
joblib.dump(detection_data, params_detection['path_save'])


## 3. BATCH FEATURE ROIS
rois_list = detection_data['detections']
features = batch_feature_rois(rois_list, params_features, params_features['path_audio'])
joblib.dump(features, params_features['path_save'])
# Write to csv
df = features_to_csv(features)
df.to_csv(params_features['path_save'][0:-7]+'.csv', index=False, sep=',')


## 4. SELECT SAMPLES FOR ANNOTATION
# Filter rois by duration
tlen_min, tlen_max = params_selrois['tlen_lims']
rois_tlen = df.max_t - df.min_t
idx_keep = (rois_tlen > tlen_min) & (rois_tlen < tlen_max)
df_sel = df.loc[idx_keep,:]

# Stratified sampling using KMeans
from sklearn.cluster import KMeans
clf = KMeans(n_clusters=params_selrois['n_clusters'], random_state=42)
X = df_sel.loc[:,df_sel.columns.str.startswith('shp')]
clf.fit(X)
df_sel.loc[:,'cluster'] = clf.labels_
# stratified sampling 
df_stratsample = df_sel.groupby('cluster', group_keys=False).apply(lambda x: x.sample(params_selrois['n_samples_per_cluster']))
df_stratsample.reset_index()
df_stratsample.to_csv(params_selrois['path_save'], index=False)


## 5. VISUALIZE CLUSTERS
from sklearn import manifold
import matplotlib.pyplot as plt
import seaborn as sns
X = df_sel.loc[:,df_sel.columns.str.startswith('shp')]
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, 
                     n_iter=5000, verbose=1, perplexity=50)
Y = tsne.fit_transform(X)
# plot data
plt.close('all')
sns.scatterplot(Y[:,0], Y[:,1], hue=df_sel.cluster, alpha=0.2, 
                palette=sns.color_palette('dark', n_colors=12))
sns.despine()