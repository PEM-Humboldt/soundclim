#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:58:16 2020

@author: juan
"""

import joblib
import pandas as pd

path_trainds = './trainds/trainds.joblib'
path_mannot = './trainds/trainds_mannot.txt'
path_save = './trainds/trainds_features_mannot.csv'


# load data
train_data = joblib.load(path_trainds)
gt = pd.read_csv(path_mannot, header=None, usecols=[0,1,2], sep='\t', 
                 names=['onset','offset','label'])

# format
"""
NOTE SE NECESITA HACER ALGO PARA REMPLAZAR TODO LO NUMÉRICO CON NA
"""
lab_wname = gt.loc[(gt.label.str[1]=='_'),'label']

# check typo errors and fix in mannot file
lab_wname.value_counts()

# 
aux = lab_wname.str.split('_')
lab_gt = aux.apply(lambda x: x[0])
lab_gt.value_counts()

# save in train_data object
