# -*- coding: utf-8 -*-
"""
Created on Tue Aug 05 09:03:40 2014

@author: c_cook
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



################## Importing data ###################

# Export sampled training data
file_name = "Raw/train (downsampled zeroes).csv"
kaggle_pth = "S:\General\Training\Ongoing Professional Development\Kaggle/"
data_pth = "Predicting fire losses with Liberty Mutual\Data/"
file_name = "Raw/train (downsampled zeroes).csv"

fire_train_smp = pd.read_csv(kaggle_pth + data_pth + file_name)
fire_test = pd.read_csv(kaggle_pth + data_pth + "Raw/test.csv")