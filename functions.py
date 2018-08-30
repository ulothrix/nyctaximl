#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ulothrix
"""
import pandas as pd
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler

# istenilen saat aralığının alınma fonksiyonu
def oglen(oglen12,oglen13,data):
    oglen_12 = pd.to_datetime(oglen12)
    oglen_13 = pd.to_datetime(oglen13)
    oglen_1213 = (data['tpep_pickup_datetime']>oglen_12) & (data['tpep_pickup_datetime']<oglen_13)
    return data.loc[oglen_1213]

def alloglen(d1,d2,d3):
    alldata = pd.concat([d1,d2,d3])
    return alldata

def reindexing(d1):
    d1 = d1.reset_index()
    t_pindex = d1.drop('index',axis=1)
    d1 = t_pindex
    d1.index.name = 'index'
    return d1

def splitDateTime(d1):
    temp = pd.DatetimeIndex(d1['tpep_pickup_datetime'])
    d1['Time'] = temp.time
    d1['Date'] = temp.date
    d1 = d1[['VendorID','Date','Time','tpep_pickup_datetime','tpep_dropoff_datetime','passenger_count','trip_distance','RatecodeID','store_and_fwd_flag','PULocationID','DOLocationID','payment_type','fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge','total_amount']]
    return d1

# Sınıf sayısı için 1 ve 2 olan değerleri 0 ve 1 yapmalıyız ki iki farklı sınıfımız olduğunu Estimator API anlasın
def label_fix(label):
    if label == 1:
        return 0
    else:
        return 1

def label_keras_fix(label):
    if label == 'N':
        return 0
    else:
        return 1
    
def scaling(train_x,test_x,col_to_norm):
    scaler = MinMaxScaler()
    train_x[col_to_norm] = scaler.fit_transform(train_x[col_to_norm])
    test_x[col_to_norm] = scaler.fit_transform(test_x[col_to_norm])
    

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')