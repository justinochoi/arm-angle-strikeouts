#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 20:19:35 2025

@author: justinchoi
"""

import pandas as pd 
import numpy as np
from sklearn.ensemble import IsolationForest 
from pygam import LinearGAM, s 

pd.set_option('display.max_columns', None) 

# read in data
train = pd.read_csv('/Users/justinchoi/Downloads/nwds-k/train.csv')
test = pd.read_csv('/Users/justinchoi/Downloads/nwds-k/test.csv') 

# only two strikes 
train = train[train['strikes'] == 2]

# drop unnecessary columns
# I prefer to be pitch type-agnostic, this might be hurting me but I can live with that 
train.drop(columns=['pitch_name','pitch_type','is_strike','strikes'], inplace=True) 
test.drop(columns=['pitch_name','pitch_type','strikes'], inplace=True) 

def basic_features(df): 

    df['inning_topbot'] = df['inning_topbot'].map({'Top':1,'Bot':0}) 
    
    df['zone_length'] = df['sz_top'] - df['sz_bot'] 
    
    df['pfx_x'] = np.where(df['p_throws'] == 'L', df['pfx_x'].mul(-1), df['pfx_x'])
    
    df['release_pos_x'] = np.where(df['p_throws'] == 'L', df['release_pos_x'].mul(-1), df['release_pos_x']) 
    
    df['spin_axis'] = np.where(df['p_throws'] == 'L', df['spin_axis'].mul(-1), df['spin_axis'])

    df['platoon_adv'] = np.where(df['stand'] == df['p_throws'], 1, 0)
    
    df['movement_angle'] = np.abs(np.rad2deg(np.arctan2(df['pfx_x'], df['pfx_z']))) 
    
    df['movement_mag'] = (df['pfx_x'].pow(2) + df['pfx_z'].pow(2)).pow(0.5)
    
    df['angle_diff'] = df['movement_angle'] - df['arm_angle']
    
    for col in ['on_3b','on_2b','on_1b']: 
        df[col] = df[col].astype(int) 
    
    for col in ['stand','p_throws']: 
        df[col] = df[col].map({'R':1,'L':0}) 
    
    return df 

train = basic_features(train) 
test = basic_features(test)

# isolation forest to use 
iforest = IsolationForest(n_estimators = 200, max_samples = 'auto') 

# how unique is a pitch w/r/t movement, arm angle, and velo 
def calculate_outlier_scores(df): 
    iforest.fit_predict(df[['pfx_x','pfx_z','arm_angle','release_speed']]) 
    score = iforest.decision_function(df[['pfx_x','pfx_z','arm_angle','release_speed']]) 
    df['outlier_score'] = score
        
    return df 

train = calculate_outlier_scores(train) 
test = calculate_outlier_scores(test) 

# calculate 'unexpectedness' of movement based on arm angle using GAM 
def arm_angle_gam(df): 
    for col in ['pfx_x','pfx_z']: 
        X = df[['release_speed','arm_angle']]
        y = df[col] 
        gam = LinearGAM(s(0) + s(1)).fit(X, y) 
        df[f"{col}_resid"] = df[col] - gam.predict(X) 
        
    return df 

train = arm_angle_gam(train)
test = arm_angle_gam(test)

# split training and testing data into swings and takes 
train_swing = train[train['bat_speed'].isna() == False] 
test_swing = test[test['bat_speed'].isna() == False] 
train_take = train[train['bat_speed'].isna() == True] 
test_take = test[test['bat_speed'].isna() == True] 

def swing_features(df): 
    df['speed_length_ratio'] = df['bat_speed'] / (df['swing_length'] + 1) 
    
    df['mov_length_ratio'] = df['swing_length'] / (df['movement_mag'] + 1) 
    
    return df 

train_swing = swing_features(train_swing) 
test_swing = swing_features(test_swing)

from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import log_loss 
import catboost as cb 
import optuna 
 
# let's work on swings first 
X_swing = train_swing.drop(columns=['index','k'], axis=1)  
y_swing = train_swing['k'] 

# catboost version
def cb_objective_swing(trial): 
    params = {
        'iterations': 500, 
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True), 
        "depth": trial.suggest_int("depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=711) 
    cv_scores = [] 
    
    for train_idx, val_idx in cv.split(X_swing, y_swing): 
        X_swing_train, X_swing_val = X_swing.iloc[train_idx], X_swing.iloc[val_idx] 
        y_swing_train, y_swing_val = y_swing.iloc[train_idx], y_swing.iloc[val_idx]  
    
        model = cb.CatBoostClassifier(**params, random_seed=711) 
        model.fit(X_swing_train, y_swing_train, verbose=0) 
        val_preds = model.predict_proba(X_swing_val)[:, 1] 
        ll_score = log_loss(y_swing_val, val_preds) 
        cv_scores.append(ll_score)
    
    return np.mean(cv_scores)

swing_study = optuna.create_study(direction='minimize') 
swing_study.optimize(cb_objective_swing, n_trials=30)

best_swing_params = swing_study.best_params 

full_swing_model = cb.CatBoostClassifier(
    iterations = 500, 
    learning_rate = best_swing_params['learning_rate'], 
    depth = best_swing_params['depth'], 
    subsample = best_swing_params['subsample'], 
    colsample_bylevel = best_swing_params['colsample_bylevel'], 
    min_data_in_leaf = best_swing_params['min_data_in_leaf']
    )

full_swing_model.fit(X_swing, y_swing) 

swing_importance = {X_swing.columns[i]: full_swing_model.feature_importances_[i] 
                    for i in range(len(X_swing.columns))} 

# moving onto takes 
# bat speed and swing length are no longer features 
X_take = train_take.drop(columns=['index','k','bat_speed','swing_length'], axis=1) 
y_take = train_take['k'] 

# catboost version 
def cb_objective_take(trial): 
    params = {
        'iterations': 500, 
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True), 
        "depth": trial.suggest_int("depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=711) 
    cv_scores = [] 
    
    for train_idx, val_idx in cv.split(X_take, y_take): 
        X_take_train, X_take_val = X_take.iloc[train_idx], X_take.iloc[val_idx] 
        y_take_train, y_take_val = y_take.iloc[train_idx], y_take.iloc[val_idx]  
    
        model = cb.CatBoostClassifier(**params, random_seed=711) 
        model.fit(X_take_train, y_take_train, verbose=0) 
        val_preds = model.predict_proba(X_take_val)[:, 1] 
        ll_score = log_loss(y_take_val, val_preds) 
        cv_scores.append(ll_score)
    
    return np.mean(cv_scores)

take_study = optuna.create_study(direction='minimize') 
take_study.optimize(cb_objective_take, n_trials=30) 

best_take_params = take_study.best_params

full_take_model = cb.CatBoostClassifier(
    iterations = 500, 
    learning_rate = best_take_params['learning_rate'], 
    depth = best_take_params['depth'], 
    subsample = best_take_params['subsample'], 
    colsample_bylevel = best_take_params['colsample_bylevel'], 
    min_data_in_leaf = best_take_params['min_data_in_leaf']
    )
full_take_model.fit(X_take, y_take)

take_importance = {X_take.columns[i]: full_take_model.feature_importances_[i] 
                    for i in range(len(X_take.columns))} 


# preeictions on swings in test data
test_swing['k'] = full_swing_model.predict_proba(
    test_swing.drop(columns='index', axis=1))[:, 1] 

# predictions on take in test data 
test_take['k'] = full_take_model.predict_proba(
    test_take.drop(columns=['index','bat_speed','swing_length'], axis=1))[:, 1]


# aggregrate results, then sort by index to ensure correct submission 
all_test = pd.concat([test_swing, test_take]) 
all_test = all_test.sort_index() 

sample_sol = pd.read_csv('/Users/justinchoi/Downloads/nwds-k/sample_solution.csv') 
sample_sol['k'] = all_test['k'] 
sample_sol.to_csv('arm_angle_submission_v12.csv', index=False)








