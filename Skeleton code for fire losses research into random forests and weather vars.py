# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 09:05:40 2014

@author: p_cohen

Things to do: deal with geodem and weather vars (are they useful?), try a two-
stage (p(claim)*E[claim_size|claim)] estimated separately)
"""

# import libraries that will be useful (superset of libraries needed for code)
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.ridge import RidgeCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import time
import statsmodels as sm
from scipy.optimize import minimize
import pybrain

def weighted_gini(act,pred,weight):
    df = pd.DataFrame({"act":act,"pred":pred,"weight":weight})    
    df.sort('pred',ascending=False,inplace=True)        
    df["random"] = (df.weight / df.weight.sum()).cumsum()
    total_pos = (df.act * df.weight).sum()
    df["cum_pos_found"] = (df.act * df.weight).cumsum()
    df["lorentz"] = df.cum_pos_found / total_pos
    df["gini"] = (df.lorentz - df.random) * df.weight  
    return df.gini.sum()

def normalized_weighted_gini(act,pred,weight):
    return weighted_gini(act,pred,weight) / weighted_gini(act,act,weight)

# create lists of all the variable names in a particular class of variables     
def var_list(var_list, list_nm, num_start, num_end):
    for x in range(num_start,num_end+1):
        var_list.append(list_nm + str(object=x))

# replace missing values with median of variable
def replace_missings(data):
    for var_type in numeric_cols
        for cols in var_type:
            data[cols].fillna(value=data[cols].median(), inplace=True)

# write predictions into all samples
def write_preds_allsamps(clf, name, feats):
    samps = (fire_train_VAL_smp,  fire_train_TRAIN_smp, fire_test)
    for samp in samps:
        samp[name] = clf.predict(samp[feats])
# write predictions into all samples using predict_proba method (certain clfs)      
def write_preds_allsamps_proba(clf, name, feats):
    samps = (fire_train_VAL_smp,  fire_train_TRAIN_smp, fire_test)
    for samp in samps:
        samp[name] = clf.predict_proba(samp[feats])[:,1]

 
################## Importing data ###################

# Export sampled training data
file_name = "Raw/train (downsampled zeroes).csv"
kaggle_pth = "S:\General\Training\Ongoing Professional Development\Kaggle/"
data_pth = "Predicting fire losses with Liberty Mutual\Data/"
file_name = "Raw/train (downsampled zeroes).csv"

fire_train_smp = pd.read_csv(kaggle_pth + data_pth + file_name)
fire_test = pd.read_csv(kaggle_pth + data_pth + "Raw/test.csv")

###################### Data cleaning ################################
#handy list with each item being the name of a weatherVar
          
weatherVar = []             
var_list(weatherVar, "weatherVar", 1, 236) 
weatherVar
geodemVar = [] 
var_list(geodemVar, "geodemVar", 1, 37) 
            
# maybe remove zeroes from weather vars and change to median
numeric_cols = (weatherVar, geodemVar)       
replace_missings(fire_train_smp)
replace_missings(fire_test)

# creating a binary target to run binary classifiers
fire_train_smp['bin_target'] = 0
fire_train_smp['bin_target'][fire_train_smp['target']>0] = 1  

##################### Split train and validation ###################
# separate training data into training and validation
np.random.seed(42)
split_rate = .2
rndm_vect = np.array(np.random.rand(len(fire_train_smp.target),1))
fire_train_TRAIN_smp = fire_train_smp[rndm_vect>split_rate ]
fire_train_VAL_smp = fire_train_smp[rndm_vect<=split_rate ]

##################### Prep for classifiers ############################
        
# define features for forest
forest_feats = []
var_types_for_forest = (weatherVar, geodemVar)
for var_type in var_types_for_forest:
    for cols in var_type:
        forest_feats.append(cols) 
        
##################### Run classifiers ############################
weights = np.array([fire_train_TRAIN_smp['var11']]).squeeze()
# Forest
rndm_forest = RandomForestClassifier(n_estimators=300,
                                     min_samples_split=35,
                                     min_samples_leaf=8)
rndm_forest.fit(fire_train_TRAIN_smp[forest_feats], 
                fire_train_TRAIN_smp.bin_target, weights)
write_preds_allsamps_proba(rndm_forest, "fin_forst_preds", forest_feats)

##################### Evaluate model ##############################
models = ("fin_forst_preds", )
for mod in models:
    print mod
    normalized_weighted_gini(fire_train_VAL_smp['target'],
                             fire_train_VAL_smp[mod],
                             fire_train_VAL_smp['var11'])


    

 






