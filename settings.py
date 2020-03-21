#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Settings file to tune and save detector

SPECIES: Scinax hayii
SITE: BETANIA

@author: jsulloa
"""

path_audio = {'train' : '/Users/jsulloa/Dropbox/PostDoc/Soundclim/audio_sites/BETANIA/train/',
              'test' : '/Users/jsulloa/Dropbox/PostDoc/Soundclim/audio_sites/BETANIA/test/',
              'template': '/Users/jsulloa/Dropbox/PostDoc/Soundclim/audio_sites/BETANIA/templates/BETA-_20161006_002000_section.wav'}

detection = {'flims': (1000, 4000),
             'tlen': 0.30,
             'th': 1e-4,
             'path_save': './features/detections_shayii.joblib'}

features = {'flims' : (1000, 4000),
            'sample_rate': 22050,
            'opt_spec': {'nperseg': 512, 'overlap': 0.5, 'db_range': 250},
            'opt_shape_str': 'med',
            'path_save': './features/features_med.joblib'}

selrois = {'tlen_lims': (0.2, 0.5),
           'n_clusters': 20,
           'n_samples_per_cluster': 100,
           'path_save': './trainds/df_stratsample.csv'}

trainds = {'flims': (900,5000),
           'wl': 2,
           'path_df': './trainds/df_stratsample.csv',
           'path_save': './trainds/'}