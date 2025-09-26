# -*- coding: utf-8 -*-
"""
Generate missing completely at random (MCAR)

@author: Kevin S. Xu
"""

import numpy as np

def generate_mcar_data(data, missing_rate, missing_value=np.nan,
                       random_state=1):
    
    # Generate uniform random numbers and choose entries smaller than
    # missing_rate to be missing using a mask
    rng = np.random.default_rng(random_state)
    random_missing = rng.random((data.shape[0], data.shape[1] - 1))
    mask = np.where(random_missing < missing_rate, 1, 0)
    mask_with_label = np.hstack((mask, np.zeros((data.shape[0], 1))))
    
    data_missing = data.copy()
    data_missing[mask_with_label == 1] = missing_value
    return data_missing
