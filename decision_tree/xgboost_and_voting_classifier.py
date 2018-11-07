
# coding: utf-8

# ## XGBoost
# The hype. So I wanna join in and try the hype
# 
# ## VotingClassifier (Soft and Hard)
# It is the ultimate classifier for classical machine learning methods so why not try

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


# Input data before feature selection
input_data_before_fs = pd.read_csv('processed_train.csv', index_col=0)

# Input data after feature selection
input_data_after_fs = pd.read_csv('processed_train_after_feature.csv', index_col=0)

# Upsampling without feature selection

# Upsampling with feature selection

# Downsampling without feature selection

# Upsampling with feature selection


# List of all the input data
input_all = {
    "normal_before_fs" : input_data_before_fs,
#     "normal_after_fs" : input_data_after_fs
}


# Functions
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

def metric_consolidation(input_all, classifier, method = "cross_validation"):
    metrics = {'accuracy': 'accuracy',
               'roc_auc': make_scorer(multiclass_roc_auc_score, average='weighted'),
               'f1_weighted': 'f1_weighted'
              }
    
    for input_name, input_data in input_all.items():
        # split the data
        x_train, x_test, y_train, y_test = preprocessing(input_data)

        # fit the classifier to the training data
        classifier.fit(x_train, y_train)

        # apply all metrics to the classifier for cross_validation
        if method == "cross_validation":
            scores = tenfold(classifier, x_train, y_train, metric = metrics)
            print ("Metrics for %s: \n" %input_name)
            for metric in metrics:
                test_score_name = "test_" + metric
                test_score = scores[test_score_name]
                print ("%s Test Score: %0.2f +/- %0.2f" %(metric, test_score.mean()*100,
                                               test_score.std()*100))   
            print ("\n")
            
        if method == "test":
            y_pred = classifier.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            roc_score = multiclass_roc_auc_score(y_test, y_pred, average='weighted')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            metric_values = {'accuracy': accuracy,
                             'roc_auc': roc_score,
                             'f1_weighted': f1_weighted
                            }
            for metric in metrics:
                test_score = metric_values[metric]
                print ("%s Test Score: %0.2f +/- %0.2f" %(metric, test_score.mean()*100,
                                               test_score.std()*100)) 
            
def preprocessing(data):
    #Split data into variables types - boolean, categorical, continuous, ID
    bool_var = list(data.select_dtypes(['bool']))
    cont_var = list(data.select_dtypes(['float64']))
    cat_var = list(data.select_dtypes(['int64']))

    #Input Data can be from all except id details
    final_input_data = data[cat_var + cont_var + bool_var]
    
    x = final_input_data.loc[:, final_input_data.columns != 'Target'].values
    y = final_input_data['Target'].values
    y=y.astype('int')
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, 
                                                    random_state = 100 , stratify = y)
    
    return x_train, x_test, y_train, y_test

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import cross_validate

def tenfold(model, x, y, metric='accuracy'):
    kfold = StratifiedKFold(n_splits=10, random_state=100, shuffle=True)
    scores = cross_validate(model, x, y, cv=kfold, scoring=metric, 
                            return_train_score=True)
    return scores

# accuracy_mean = scores['test_score'].mean()
# accuracy_std = scores['train_score'].std()


# XGBoost

# Voting Classifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier

# clf1 = svm.SVC(C=0.1,decision_function_shape='ovo', kernel='linear', max_iter=-1, random_state=100)
# clf2 = RandomForestClassifier(n_estimators=50, random_state=0)
# clf3 = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#                                          max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
#                                          min_samples_split=50, min_weight_fraction_leaf=0.0,
#                                          presort=False, random_state=100, splitter='best')
# clf4 = LogisticRegression(random_state=100, penalty = 'l1', C = 10**-1)
# clf5 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
#                                          random_state=0)


# eclf = VotingClassifier(estimators=[('svm', clf1), ('rf', clf2), ('dt', clf3), ('lr', clf4), 
#                                     ('bgt', clf5)], 
#                         voting='hard')

# Voting Classifier Parameters

voting_values = ['hard', 'soft']

for voting in voting_values:
    # clf1 = svm.SVC(C=0.1,decision_function_shape='ovo', kernel='linear', max_iter=-1, random_state=100)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=0)
    clf3 = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                                             max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
                                             min_samples_split=50, min_weight_fraction_leaf=0.0,
                                             presort=False, random_state=100, splitter='best')
    clf4 = LogisticRegression(random_state=100, penalty = 'l1', C = 10**-1)
    clf5 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                             random_state=0)
    clf6 = BaggingClassifier(n_estimators=100, random_state=100, base_estimator=LogisticRegression())


    eclf = VotingClassifier(estimators=[('rf', clf2) , ('lr', clf4), 
                                        ('bgt', clf5)], voting=voting)
    
    print ("For Voting Classifier with: \n voting: %s \n" %(voting))
    metric_consolidation(input_all, eclf)
# ('dt', clf3)
# ('svm', clf1)



