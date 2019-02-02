# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 21:59:00 2019

@author: cidmedeiros
"""

def featureScaling(arr):
    from sklearn.preprocessing import MinMaxScaler
    ans_arr = []
    for i in arr:
        ans_arr.append([i])  
    scaler = MinMaxScaler()
    rescaled_arr = scaler.fit_transform(ans_arr)
    return rescaled_arr

data = [115, 140, 175]

a = featureScaling(data)
