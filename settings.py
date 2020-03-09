#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Settings file to tune and save detector

SPECIES: Scinax hayii
SITE: BETANIA

@author: jsulloa
"""

# Find ROIs
def load_compile_dataset():
    params_detection = {'flims': (1000, 4000),
                        'tlen': 0.30,
                        'th': 1e-4,
                        'path_template': '/media/juan/data/Dropbox/PostDoc/Soundclim/audio_sites/BETANIA/templates/BETA-_20161006_002000_section.wav',
                        'path_audio': '/media/juan/data/Dropbox/PostDoc/Soundclim/audio_sites/BETANIA/train/',
                        'path_save': './features/detections_shayii.joblib'}
    
    
    # Feature ROIs
    params_features = {'flims' : (1000, 4000),
                       'opt_spec': {'wl': 512, 'ovlp': 0.5, 'db_range': 250},
                       'opt_shape_str': 'med',
                       'path_audio': '/media/juan/data/Dropbox/PostDoc/Soundclim/audio_sites/BETANIA/train/',
                       'path_save': './features/features_med.joblib'}
    
    
    # Select ROIs
    params_selrois = {'tlen_lims': (0.2, 0.5),
                      'n_clusters': 12,
                      'n_samples_per_cluster': 50,
                      'path_save': './trainds/df_stratsample.csv'}

    
    return params_detection, params_features, params_selrois