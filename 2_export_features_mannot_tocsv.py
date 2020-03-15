#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:58:16 2020

@author: juan
"""

import joblib
import pandas as pd
import numpy as np

# -- Set variables --
path_trainds = './trainds/trainds.joblib'
path_mannot = './trainds/trainds_mannot.txt'
path_save = './trainds/trainds_features_mannot.csv'


# load data
train_data = joblib.load(path_trainds)
gt = pd.read_csv(path_mannot, header=None, usecols=[0,1,2], sep='\t', 
                 names=['onset','offset','label'])

# format labelling and chek typo errors
idx_annotated = (gt.label.str[1]=='_')
lab_wname = gt['label']
lab_wname.loc[~idx_annotated] = np.nan

# check typo errors and fix in mannot file
lab_wname.value_counts()

# get binary labels
lab_bin = lab_wname.str[0]
lab_bin.value_counts()

# include annotations in train dataset
lab_bin.name = 'lab_gt'
lab_wname.name = 'lab_wname'
df = pd.concat([train_data['roi_info'],
                train_data['shape_features'],
                lab_wname,
                lab_bin],
                axis=1)

df.reset_index(inplace=True)
df.to_csv(path_save, index=False)

## save in train_data object
#train_data['label'] = lab_wname
∫