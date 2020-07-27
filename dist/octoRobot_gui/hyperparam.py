# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:11:12 2020

@author: Gajanan Thenge
"""
from hyperopt import hp
from sklearn.ensemble import  RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from xgboost import XGBClassifier

param_spaces = hp.choice('classifier_type', [
    {
        'model': 'naive_bayes',
        'alpha': hp.uniform('alpha', 0.0, 2.0),
        'fit_prior':hp.choice('fit_prior', [True, False])
    },
    {
        'model': "knn",
        'n_neighbors': hp.choice('n_neighbors', range(1, 30)),
        'weights': hp.choice('weights', ['uniform', 'distance']),

    },
    {
        'model': 'logistic_regression',
        'C': hp.uniform('C_logi', 0.0, 10.0),
        'solver': hp.choice('solver',
                            ["newton-cg",
                             'sag', "lbfgs"]),
        'max_iter': hp.choice('max_iter', range(50, 1000)),
        'class_weight': hp.choice('class_weight', ['balanced']),
        'random_state': hp.choice('random_state_log', [12345])
        
    },

    {
        'model': 'random_forest',
        'max_depth': hp.choice('max_depth_rf', range(1, 10)),
        'max_features': hp.choice('max_features_rf', ["sqrt", 'log2']),
        'n_estimators': hp.choice('n_estimators_rf', range(1, 100)),
        'criterion': hp.choice('criterion_rf', ["gini", "entropy"]),
#        'bootstrap': hp.choice('bootstrap', [True]),
        'random_state': hp.choice('random_state', [12345]),
#        'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 5)),
#        'min_samples_split': hp.choice('min_samples_split', range(5, 10))
    },
    {
        'model': 'xgboost',
        'max_depth': hp.choice('max_depth_xg', range(3, 6)),
        'n_estimators': hp.choice('n_estimators_xg', range(30, 100)),
        'gamma': [0,1,5],
        'scale_pos_weight':[1,10,25,50,75,99,100],
        #'subsample': [0.4,0.6,0.8],
        #'colsample_bytree': [0.5,0.6,0.8]
    },    
])

parameter_dict = {
    'random_forest':
        {
            'clf': [RandomForestClassifier(random_state=12345,n_jobs=3)],
            'clf__max_features': ['sqrt', 'log2'],
            'clf__max_depth': [3, 5, 6, 7, 8, 9, 10],
            'clf__n_estimators': [20, 30, 40, 50]
            # more parameters for RandomForesetClassifier
        },
    'naive_bayes':
        {
            'clf': [BernoulliNB()],
            'clf__alpha': [1.0, 2.0],
        },
    'logistic_regression':
        {
            'clf': [LogisticRegression(random_state=12345)],
            'clf__penalty': ['l2'],
            'clf__C': np.logspace(0, 4, 8)
        },
    'knn':
        {
            'clf': [KNeighborsClassifier(n_jobs=3)],
            'clf__n_neighbors': [5, 10, 20, 25, 30],
            'clf__weights': ['uniform', 'distance'],
        },
    'xgboost':
        {
            'clf': [XGBClassifier(random_state=12345,n_jobs=3)],
            'clf__max_depth': [3,4,6,8],
            'clf__n_estimators': [30,50,80,100],
            'clf__gamma': [0,1,5],
            'clf__scale_pos_weight':[1,10,25,50,75,99,100],            
            #'clf__subsample': [0.4,0.6,0.8],
            #'clf__colsample_bytree': [0.5,0.6,0.8],
        }
    #    'svm' : {
    #        'clf': [SVC()],
    #        'clf__C': [0.5, 1, 3],
    #        'clf__kernel': ['rbf', 'linear', 'poly']
    #    }
}
