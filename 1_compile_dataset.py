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
import settings
from soundclim_utilities import (features_to_csv, 
                                 batch_find_rois, 
                                 batch_feature_rois, 
                                 listdir_pattern)

## 1. CALIBRATE FIND ROIS
s, fs = sound.load(settings.path_audio['template'])
rois = find_rois_cwt(s, fs, 
                     flims = settings.detection['flims'], 
                     tlen = settings.detection['tlen'], 
                     th = settings.detection['th'], 
                     display=True, figsize=(13,6))
print(rois)


## 2. BATCH FIND ROIS
flist = listdir_pattern(settings.path_audio['train'], ends_with='.wav')
flist = pd.DataFrame(flist, columns=['fname'])
detection_data = batch_find_rois(flist, settings.detection, 
                                 settings.path_audio['train'])

## 3. BATCH FEATURE ROIS
rois_list = detection_data['detections']
features = batch_feature_rois(rois_list, settings.features, settings.path_audio['train'])
joblib.dump(features, settings.features['path_save'])
# Write to csv
df = features_to_csv(features)
df.to_csv(settings.features['path_save'][0:-7]+'.csv', index=False, sep=',')


## 4. SELECT SAMPLES FOR ANNOTATION
# Filter rois by duration
tlen_min, tlen_max = settings.selrois['tlen_lims']
rois_tlen = df.max_t - df.min_t
idx_keep = (rois_tlen > tlen_min) & (rois_tlen < tlen_max)
df_sel = df.loc[idx_keep,:]

# Stratified sampling using KMeans. 
from sklearn.cluster import KMeans
clf = KMeans(n_clusters=settings.selrois['n_clusters'], random_state=42)
X = df_sel.loc[:,df_sel.columns.str.startswith('shp')]
clf.fit(X)
df_sel.loc[:,'cluster'] = clf.labels_
# stratified sampling 
n_max = settings.selrois['n_samples_per_cluster']
gby = df_sel.groupby('cluster', group_keys=False)
# if cluster is smaller than n_max, take all samples from cluster
df_stratsample = gby.apply(lambda x: x.sample(n= n_max, random_state=42) if len(x)> n_max else x)
df_stratsample.reset_index()
df_stratsample.to_csv(settings.selrois['path_save'], index=False)


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


## 6. FORMAT TRAINING DATASET
from soundclim_utilities import format_trainds
import numpy as np
from scipy.io.wavfile import write
df = pd.read_csv(settings.trainds['path_df'])
train_data = format_trainds(df, 
                            settings.trainds['flims'], 
                            settings.trainds['wl'], 
                            settings.path_audio['train'])

# write joblib object with all data
joblib.dump(train_data, settings.trainds['path_save']+'trainds.joblib')
# write audio file
sx = np.concatenate(train_data['audio'], axis=0)
write(settings.trainds['path_save']+'trainds.wav', 22050, sx)
# write annotations
seg = train_data['segments']
label_fmt = [i + '-'+ j for i, j in zip(train_data['maad_label'].astype(str), 
                                        np.arange(seg.shape[0]).astype(str))]
audacity_annot = pd.DataFrame({'onset': np.arange(0, seg.shape[0])*settings.trainds['wl'] + seg.onset, 
                               'offset': np.arange(0, seg.shape[0])*settings.trainds['wl'] + seg.offset,
                               'label': label_fmt})
audacity_annot.to_csv(settings.trainds['path_save']+'trainds.txt', 
                      sep='\t', index=False, header=False)
