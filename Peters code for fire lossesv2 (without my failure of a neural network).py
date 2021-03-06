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
from sklearn.ensemble import  RandomForestRegressor, AdaBoostRegressor
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.ridge import RidgeCV
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import svm
import numpy as np
import time
import statsmodels as sm
from scipy.optimize import minimize
from __future__ import division

def weighted_gini(act,pred,weight):
    df = pd.DataFrame({"act":act,"pred":pred,"weight":weight}) 
    df = df.sort('pred',ascending=False) 
    df["random"] = (df.weight / df.weight.sum()).cumsum()
    total_pos = (df.act * df.weight).sum()
    df["cum_pos_found"] = (df.act * df.weight).cumsum()
    df["lorentz"] = df.cum_pos_found / total_pos
    n = df.shape[0]
    #df["gini"] = (df.lorentz - df.random) * df.weight 
    #return df.gini.sum()
    gini = sum(
    df.lorentz[1:].values * (df.random[:-1])) - sum(
    df.lorentz[:-1].values * (df.random[1:]))
    return gini

def normalized_weighted_gini(act,pred,weight):
    return weighted_gini(act,pred,weight) / weighted_gini(act,act,weight)

# create lists of all the variable names in a particular class of variables     
def var_list(var_list, list_nm, num_start, num_end):
    for x in range(num_start,num_end+1):
        var_list.append(list_nm + str(object=x))

# replace missing values with median of variable
def replace_missings(data):
    for var_type in numeric_cols:
        for cols in var_type:
            data[cols].fillna(value=data[cols].median(), inplace=True)
            
# summarize the variable selection process using lasso regression
def selection_summary(var_list, lassoed_list, class_):
  first =  "Out of " + str(object=len(var_list)) + " starting " + class_ 
  second =  " vars, "  + str(object=len(lassoed_list)) + " were chosen"  
  print first + second
          
def cat_to_allstr(var):
    fire_train_smp[var] = fire_train_smp[var].astype(str)
    fire_test[var] = fire_test[var].astype(str)
    
# use lasso regression to select only useful variables from a lon list
def varselect_w_lass(all_vars_list, selected_vars, alpha_val):
    lass = Lasso(alpha=alpha_val,
                 positive=True, max_iter=100000 , tol=.0001)
    lass.fit(np.array(fire_train_TRAIN_smp[all_vars_list]),
             np.array(fire_train_TRAIN_smp.target ))
    for x in range(1, len(all_vars_list)):
        if lass.coef_[x]> .00000001:
            selected_vars.append(all_vars_list[x])
            
# turn categorical variables into true false dummies for each level       
def add_dummies(level_var, list_nm, data):
    # this function creates a variable that represents the is_exciting probability
    # at each level of a categorical variable
    levels = fire_train_smp.groupby(level_var).groups.keys()
    for level in levels:
        new_var_nm = "is_" + str(object=level_var) + "_" + str(object=level)
        data[new_var_nm ]=0
        data[new_var_nm][data[level_var]==level] = 1  

# store a list of the names of the dummies that get created in previous func    
def write_dum_lists(level_var, list_nm, data):
    # stores a list of all but one of the dummies for a cat var
    levels = fire_train_smp.groupby(level_var).groups.keys()
    list_nm = []
    for level in levels:  
        new_var_nm = "is_" + str(object=level_var) + "_" + str(object=level)
        list_nm.append(new_var_nm)
    del list_nm[0]
    return list_nm

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
        
# write preds for classifier models times size predictions
def write_szpreds_allsamps(class_preds, size_preds, new_nm, scale):
    samps = (fire_train_VAL_smp,  fire_train_TRAIN_smp, fire_test)
    for samp in samps:
        samp[new_nm] = (scale * samp[size_preds] * samp[class_preds])     
        
def make_net_data(input_data, supervised_set): 
    supervised_set.setField('input', input_data[net_feats])
    y_net = np.array(input_data['target']).reshape(-1,1)
    supervised_set.setField('target', y_net)
    
# create z-scores for a variable in all samples        
def z_score_allsamps(var, var_nm):             
    samps = (fire_train_VAL_smp,  fire_train_TRAIN_smp, fire_test)
    var_mean = fire_train_TRAIN_smp[var].mean()
    var_std = fire_train_TRAIN_smp[var].std()
    for samp in samps:
        samp[var_nm] = (samp[var] - var_mean)/var_std

# create a final submission    
def write_submission(clf, sub_nm, data):
    submission= pd.DataFrame(data.id)
    submission['target'] = data[clf]
    submission.to_csv(kaggle_pth + data_pth + sub_nm, index=False)
    
def check_weight(coef_array):    
    fire_train_VAL_smp['ens_preds'] = 0
    for x in range(0, len(ens_clfs)):
        fire_train_VAL_smp['ens_preds'] += (coef_array[x] * 
                                            fire_train_VAL_smp[ens_clfs[x]])
    return 1 - normalized_weighted_gini(fire_train_VAL_smp['target'],
                fire_train_VAL_smp['ens_preds'],
                fire_train_VAL_smp['var11'])
              
def write_ensemble(data, seed):
    nm = 'ens_preds' + str(object= seed)
    data[nm] = 0
    for x in range(0, len(ens_clfs)):
        data[nm] += optimizer.x[x] * data[ens_clfs[x]]
 
################## Importing data ###################
kaggle_pth = "S:\General\Training\Ongoing Professional Development\Kaggle/"
data_pth = "Predicting fire losses with Liberty Mutual\Data/"

fire_train = pd.read_csv(kaggle_pth + data_pth + "Raw/train.csv")
fire_test = pd.read_csv(kaggle_pth + data_pth + "Raw/test.csv")
train_obs = len(fire_train.target)
# Down sample zero cost policies in train
train_positives = pd.DataFrame(fire_train.target!=0)
np.random.seed(18)
train_rnd = pd.DataFrame(np.random.rand(train_obs,1))
ones_to_keep = (train_positives.target == True)
zeroes_to_keep = (train_positives['target'] == False) & (train_rnd[0] > .85)
fire_train_smp = pd.DataFrame(fire_train.ix[(ones_to_keep) |
                             (zeroes_to_keep), :])
# Export sampled training data 
file_name = "Raw/train (downsampled zeroes).csv"
fire_train_smp.to_csv(kaggle_pth + data_pth + file_name)

######### START HERE UNLESS GOOD REASON#######
# Import so as not to re-run
kaggle_pth = "S:\General\Training\Ongoing Professional Development\Kaggle/"
data_pth = "Predicting fire losses with Liberty Mutual\Data/"
fire_test = pd.read_csv(kaggle_pth + data_pth + "Raw/test.csv")
fire_train_smp = pd.read_csv(kaggle_pth + data_pth + file_name)

###################### Data cleaning ################################
#handy lists
cont_vars = []
var_list(cont_vars, "var", 10, 17)
crimeVar = []
var_list(crimeVar, "crimeVar", 1, 9)           
geodemVar = [] 
var_list(geodemVar, "geodemVar", 1, 37)             
weatherVar = []             
var_list(weatherVar, "weatherVar", 1, 236) 
            
# maybe remove zeroes from weather vars and change to non-zero mean?
numeric_cols = (cont_vars, crimeVar, geodemVar, weatherVar)       
replace_missings(fire_train_smp)
replace_missings(fire_test)

dum_dict = {}
cat_names = ["dummy_dummies", "var1_dummies", "var2_dummies", "var3_dummies",
             "var4_dummies", "var5_dummies", "var6_dummies", "var7_dummies",
             "var8_dummies", "var9_dummies"]
cat_vars = ["dummy", "var1", "var2", "var3", "var4", "var5", "var6",  "var7",
            "var8", "var9"]
for x in range(0, len(cat_vars)):
    cat_to_allstr(cat_vars[x])
    add_dummies(cat_vars[x], cat_names[x], fire_train_smp)
    add_dummies(cat_vars[x], cat_names[x], fire_test)
    dum_dict[cat_names[x]] = write_dum_lists(cat_vars[x],
                                             cat_names[x], fire_train_smp)
                                             
# this is, okay, to be honest, not particularly smart
fire_train_smp['var7'][fire_train_smp['var7']=='Z'] = 3
fire_test['var7'][fire_test['var7']=='Z'] = 3
fire_train_smp

['var7'] = fire_train_smp['var7'].astype(int)
fire_test['var7'] = fire_test['var7'].astype(int)

# creating a binary target to run binary classifiers
fire_train_smp['bin_target'] = 0
fire_train_smp['bin_target'][fire_train_smp['target']>0] = 1  

# creating weighted loss vars
fire_train_smp['loss'] = fire_train_smp['target'] * fire_train_smp['var11'] 

# for convenience
fire_test['bin_target'] = 0
fire_test['target'] = 0

##################### Split train and validation ###################
for seed in range(329,350):
    np.random.seed(seed)
    split_rate = .35
    rndm_vect = np.array(np.random.rand(len(fire_train_smp.target),1))
    fire_train_TRAIN_smp = fire_train_smp[rndm_vect>split_rate ]
    fire_train_VAL_smp = fire_train_smp[rndm_vect<=split_rate ]
    
    ##################### Prep for classifiers ############################
    z_cont_vars = []
    for var in cont_vars:
        z_name = "z_" + var 
        z_score_allsamps(var, z_name)
        z_cont_vars.append(z_name)
    
    # Use Lasso for variable selection on big sets of vars
    lassoed_weather = []
    lassoed_geo = []
    lassoed_crime = []
    
    varselect_w_lass(weatherVar, lassoed_weather, .0000001)
    varselect_w_lass(geodemVar, lassoed_geo,  .00000001)
    varselect_w_lass(crimeVar, lassoed_crime,  .0000000001)
    
    selection_summary(geodemVar, lassoed_geo, "geodem")
    selection_summary(weatherVar, lassoed_weather, "weather")
    selection_summary(crimeVar, lassoed_crime, "crime")
    
    # define features for ridge regression
    ridge_feats = []
    var_types_for_ridge = (cont_vars, lassoed_weather, lassoed_geo, lassoed_crime,
                           dum_dict['dummy_dummies'], dum_dict['var8_dummies'],
                           dum_dict['var1_dummies'], dum_dict['var2_dummies'],
                           dum_dict['var3_dummies'], dum_dict['var4_dummies'],
                           dum_dict['var5_dummies'], dum_dict['var6_dummies'],
                           dum_dict['var9_dummies'])
    for var_type in var_types_for_ridge:
        for cols in var_type:
            ridge_feats.append(cols)
          
    # define features for forest
    forest_feats = ridge_feats
            
    # define features for ada
    ada_feats = forest_feats
    ada_feats.remove("var13")    
    
    # define features for logit
    logit_feats = ridge_feats
    
    # define neural features
    net_feats = ridge_feats
    
    # create svc data
    z_cont_vars = []
    for cont in cont_vars:
        cont_nm = "z_" + cont 
        z_score_allsamps(cont, cont_nm)
        z_cont_vars.append(cont_nm)
        
    train_obs_2 = fire_train_TRAIN_smp.target.count()
    np.random.seed(18)
    train_rnd = pd.DataFrame(np.random.rand(train_obs_2,1))
    train_positives = pd.DataFrame(fire_train_TRAIN_smp.target!=0)
    ones_to_keep = (train_positives.target == True)
    zeroes_to_keep = (train_positives.target == False) & (train_rnd[0] > .35)
    svc_train = pd.DataFrame(fire_train_TRAIN_smp.ix[(ones_to_keep) |
                             (zeroes_to_keep), :])
    svc_train.target.count()
    svc_feats = []
    var_types_for_svc = (lassoed_weather, z_cont_vars,
                           lassoed_geo, lassoed_crime,
                           dum_dict['dummy_dummies'], dum_dict['var8_dummies'],
                           dum_dict['var1_dummies'], dum_dict['var2_dummies'],
                           dum_dict['var3_dummies'], dum_dict['var4_dummies'],
                           dum_dict['var5_dummies'], dum_dict['var6_dummies'],
                           dum_dict['var9_dummies'])
    for var_type in var_types_for_svc:
        for cols in var_type:
            svc_feats.append(cols)
            
    ##################### Run classifiers ############################
    weights = np.array([fire_train_TRAIN_smp['var11']]).squeeze()
    
    # Ridge
    ## using weights crashes this, don't know why       
    ridge = RidgeCV(np.array([1.5]), store_cv_values=True, normalize=True)
    ridge.fit(fire_train_TRAIN_smp[ridge_feats], fire_train_TRAIN_smp.target)
    write_preds_allsamps(ridge, "fin_rdg_preds", ridge_feats)
    
    # claim size conditional on claim ridge
    ones_only = fire_train_TRAIN_smp['target']>0
    size_train = fire_train_TRAIN_smp.ix[ones_only, :]
    size_ridge = RidgeCV(np.array([1.5]), store_cv_values=True, normalize=True)
    size_ridge.fit(size_train[ridge_feats], size_train.target)
    write_preds_allsamps(size_ridge, "size_rdg_preds", ridge_feats)
    
    # Lasso
    lass = Lasso(alpha=.0000001, positive=True, max_iter=100000 ,
                 tol=.001, normalize=True)
    lass.fit(np.array(fire_train_TRAIN_smp[ridge_feats]),
             np.array(fire_train_TRAIN_smp.target))
    write_preds_allsamps(lass, "fin_lass_preds", ridge_feats)
    
    # Forest
    t0= time.time()
    rndm_forest = RandomForestClassifier(n_estimators=280, 
                                         min_samples_split=30,
                                         min_samples_leaf=3,
                                         verbose=1)
    rndm_forest.fit(fire_train_TRAIN_smp[forest_feats], 
                    fire_train_TRAIN_smp.bin_target, weights)
    write_preds_allsamps_proba(rndm_forest, "fin_forst_preds", forest_feats)
    print "It took {time} minutes to fit forest".format(time=(time.time()
                                                              -t0)/60)                                                    
    
    # Adaboost
    t0= time.time() 
    ada = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1, max_features=20,
                                                  min_samples_leaf=5),
                                n_estimators = 450, 
                                learning_rate=.16)
    ada.fit(fire_train_TRAIN_smp[ada_feats], 
                    fire_train_TRAIN_smp.bin_target, weights)
    write_preds_allsamps_proba(ada, "fin_ada_preds", ada_feats)
    print "It took {time} minutes to fit ada".format(time=(time.time()-t0)/60)
   
    
    # Logit
    logit = LogisticRegression()
    logit.fit(fire_train_TRAIN_smp[logit_feats], 
                    fire_train_TRAIN_smp.bin_target)
    write_preds_allsamps_proba(logit, "fin_log_preds", logit_feats)
    
    # SVM (?)
    t0= time.time() 
    svc = svm.SVC(probability=True)
    svc.fit(svc_train[svc_feats], svc_train.bin_target, svc_train.var11)
    write_preds_allsamps_proba(svc, "fin_svc_preds", svc_feats)
    print "It took {time} minutes to fit SVC".format(time=(time.time()-t0)/60)
         
    # multiply binary predictions with conditional claim size predictions
    write_szpreds_allsamps('fin_forst_preds', 'size_rdg_preds', 'forst_w_size'
                            , 1)
    write_szpreds_allsamps('fin_log_preds', 'size_rdg_preds','log_w_size' , 1)
    write_szpreds_allsamps('fin_svc_preds', 'size_rdg_preds','svc_w_size' , 1)
    write_szpreds_allsamps('fin_ada_preds', 'size_rdg_preds','ada_w_size' , 1)
    # Ensemble
    pred_nms = ["fin_rdg_preds","fin_forst_preds", "fin_ada_preds",
                "fin_log_preds", "size_rdg_preds", 'forst_w_size',
                'log_w_size', "fin_lass_preds", 'var13',
                'size_rdg_preds', 'svc_w_size', 'fin_svc_preds'] 
    z_pred_nms = ["z_rdg_preds","z_forst_preds", "z_ada_preds","z_log_preds",
                  "z_size_preds",'z_forst_w_size', 'z_log_w_size', 
                  "z_lass_preds",
                  'z_var13', 'z_size_rdg_preds', 'z_svc_w_size', 'z_svc_preds']
                            
    # , "z_net_preds"
    for x in range(0,len(pred_nms)):             
        z_score_allsamps(pred_nms[x], z_pred_nms[x])
    
    ens_clfs = ['z_rdg_preds', 'z_forst_w_size','z_ada_preds', 'z_log_w_size'
                , 'z_lass_preds', 'var13', 'z_svc_preds']
    fire_train_TRAIN_smp[ens_clfs].cov()
    # this optimizer chooses optimal weights for adding the chosen models using
    # linear weights on z-scores of predictions for each model
    optimizer = minimize(check_weight, np.array([1, 1, 1, 1,1,1,1 ]),
                         method='nelder-mead',
                         options= {'xtol':1e-1, 'disp':True})
    
    # write ensemble predictions
    
    write_ensemble(fire_train_VAL_smp, seed)
    write_ensemble(fire_test, seed)
    ridge = RidgeCV(np.array([0.1]), store_cv_values=True, normalize=True)
    ridge.fit(fire_train_VAL_smp[ens_clfs], fire_train_VAL_smp.target)
       
    name = 'ens_preds' + str(object=seed) 
    write_preds_allsamps(ridge, name, ens_clfs)
    z_name = 'z_ens_preds' + str(object=seed)
    fire_test[z_name] = (fire_test[name] - 
                         fire_test[name].mean())/fire_test[name].std()
    del fire_test[name]
    
   
    
    ##################### Evaluate model ##############################
    ens_preds_nm = 'ens_preds' + str(object=seed)
    models = ('z_rdg_preds', 'z_forst_preds' , 'z_forst_w_size' ,
              'z_ada_preds', 'ada_w_size',
              'z_log_preds', 'z_log_w_size', 'z_lass_preds',
              ens_preds_nm, 'var13', 'fin_svc_preds',
              'z_svc_w_size', 'z_size_rdg_preds') 
  
    for mod in models:
        print mod
        normalized_weighted_gini(fire_train_VAL_smp['target'],
                                 fire_train_VAL_smp[mod],
                                 fire_train_VAL_smp['var11'])
                             
##################### Create cross val predictions ##########
fire_test['ens_preds'] = 0                             
for x in range(300,337):
    nm = 'z_ens_preds' + str(object=x) 
    fire_test['ens_preds'] += fire_test[nm]                           

#################### Create Submission #############################
sub_nm = '/Submission/CV 37 folds with ridge ens.csv'  
write_submission('ens_preds', sub_nm, fire_test)

#################### Create combined submission Submission #######
kaggle_pth = "S:\General\Training\Ongoing Professional Development\Kaggle/"
data_pth = "Predicting fire losses with Liberty Mutual\Data/"
PK_path = "/Raw/28-08-14 09 35 - Third stage.csv"
PK_test = pd.read_csv(kaggle_pth + data_pth + PK_path)
PK_test['PK_preds'] = ((PK_test['target'] - PK_test['target'].mean())
                        /PK_test['target'].std())
                        
merged_test = pd.merge(fire_test,PK_test, on='id')
merged_test[['ens_preds', 'PK_preds']].corr()
merged_test['comb_pred'] = 3*merged_test['ens_preds'] + merged_test['PK_preds'] 
sub_nm = '/Submission/combined methods corrected PC upweight.csv'  
write_submission('comb_pred', sub_nm, merged_test)



    

 






