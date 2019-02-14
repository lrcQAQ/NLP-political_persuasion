from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import argparse
import sys
import os

# packages
import csv
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.model_selection import KFold
from scipy import stats


def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    accuracy = np.sum(np.diag(C)) / np.sum(C)
    return accuracy


def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recalls = np.diag(C) / np.sum(C, axis = 1)
    return recalls


def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    precs = np.diag(C) / np.sum(C, axis = 0)
    return precs
    
    
def get_classifier(i):
    
    # classifier 1: linear SVC
    if(i == 1):
        model = LinearSVC()
    
    # classifier 2: radial SVC
    if(i == 2):
        model = SVC(gamma=2, max_iter=5000)
    
    # classifier 3: random forest classifier
    if(i == 3):
        model = RandomForestClassifier(max_depth=5, n_estimators=10)
    
    # classifier 4: mlp classifier
    if(i == 4):
        model = MLPClassifier(alpha=0.05)
    
    # classifier 5: adaboost classifier
    if(i == 5):
        model = AdaBoostClassifier()
    
    return model


def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    
    # load npz data file and split
    data = np.load(filename)['arr_0']
    X_train, X_test, y_train, y_test = train_test_split(data[:, :173], data[:, 173], test_size=0.2)
    
    # initialization
    csv_content = []
    iBest = None
    best_acc = 0
    
    # train on different classifiers
    for i in range(1, 6):
        model = get_classifier(i)
        
        # train on data and get predictions
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        
        # evaluate predictions
        C = confusion_matrix(y_test, y_predict)
        acc = accuracy(C)
        rec = recall(C)
        prec = precision(C)
        
        # update iBest
        if(acc > best_acc):
            iBest = i
        
        # store result for current classifier
        curr_res = [i, acc] + list(rec) + list(prec) + C.flatten().tolist()
        csv_content.append(curr_res)
        
    # write res into csv file
    with open('a1_3.1.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerows(csv_content)
    
    return (X_train, X_test, y_train, y_test,iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
    '''
    
    # choose the best model from 3.1
    model = get_classifier(iBest)
    
    # initialization
    training_size = [1000, 5000, 10000, 15000, 20000]
    csv_content = []
    
    # for each trianing size
    for next_size in training_size:
        
        # re-sample data
        idxs = np.random.choice(X_train.shape[0], next_size, replace=False)
        X_train_sample = X_train[idxs]
        y_train_sample = y_train[idxs]
        
        # for return
        if(next_size == 1000):
            X_1k = X_train_sample
            y_1k = y_train_sample
        
        # training, prediction and evaluate accuracy
        model.fit(X_train_sample, y_train_sample)
        y_predict = model.predict(X_test)
        C = confusion_matrix(y_test, y_predict)
        curr_acc = accuracy(C)
        
        csv_content.append(curr_acc)
        
    # write accuracy into csv file
    with open('a1_3.2.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerows([csv_content])
    
    return (X_1k, y_1k)


def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    
    # initialiation
    csv_content1 = []
    csv_content2 = []
    ks = [5, 10, 20, 30, 40, 50]
    
    # process for each k feature selection
    for k in ks:
        
        # select best k features for 32k and 1k
        selector_32k = SelectKBest(f_classif, k)
        selector_1k = SelectKBest(f_classif, k)
        X_new_32k = selector_32k.fit_transform(X_train, y_train)
        X_new_1k = selector_1k.fit_transform(X_1k, y_1k)
        
        # pvalues for 32k and 1k
        pp_32k = selector_32k.pvalues_
        idx_32k = selector_32k.get_support(indices=True)
        best_k_32k = pp_32k[idx_32k]
        best_k_32k = np.sort(best_k_32k)
        
        pp_1k = selector_1k.pvalues_
        idx_1k = selector_1k.get_support(indices=True)
        best_k_1k = pp_1k[idx_1k]
        best_k_1k = np.sort(best_k_1k)
        
        # line 1 - 6, for 32k only
        curr_line_32k = [k]
        curr_line_32k.extend(best_k_32k)
        csv_content1.append(curr_line_32k)
        
        # fit using the best classifier on 32k and 1k with 5 features
        if(k == 5):
            # get best classifier
            model = get_classifier(i)
            
            # train, predict and get accuracy
            acc = []
            model_1k = model
            model_32k = model
            
            # transform training data to 5 features
            X_1k_5 = selector_1k.transform(X_1k)
            X_train_5 = selector_32k.transform(X_train)
            X_test_1k = selector_1k.transform(X_test)
            X_test_32k = selector_32k.transform(X_test)
            
            # train on 1k
            model_1k.fit(X_1k_5, y_1k)
            y_predict_1k = model_1k.predict(X_test_1k)
            C_1k = confusion_matrix(y_test, y_predict_1k)
            acc_1k = accuracy(C_1k)
            acc.append(acc_1k)
            
            # train on 32k
            model_32k.fit(X_train_5, y_train)
            y_predict_32k = model_32k.predict(X_test_32k)
            C_32k = confusion_matrix(y_test, y_predict_32k)
            acc_32k = accuracy(C_32k)
            acc.append(acc_32k)
            
            # store result
            csv_content2.append(acc)
    
    # write content to csv file
    with open('a1_3.3.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerows(csv_content1)
        csvwriter.writerows(csv_content2)


def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    
    # load data
    data = np.load(filename)['arr_0']
    csv_content1 = []
    csv_content2 = []
    
    # KFold
    kf = KFold(n_splits=5, shuffle=True)
    
    # for each classfier
    for j in range(1, 6):
        
        model = get_classifier(j)
        X_all = data[:, :173]
        y_all = data[:, 173]
        curr_acc = []
        
        # doing cross validation
        for train_idx, test_idx in kf.split(X_all):
            # split training and testing data
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]
            
            # fit model and evaluate accuracy
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            C = confusion_matrix(y_test, y_predict)
            acc = accuracy(C)
            curr_acc.append(acc)
        
        # store all 5 KFold result for current classifier
        csv_content1.append(curr_acc)
    
    # find pvalues of best classifier against other classifiers
    acc_from_best = csv_content1[i-1]
    for next_acc in csv_content1:
        if(next_acc != acc_from_best):
            S = stats.ttest_rel(acc_from_best, next_acc)
            # only store the pvalue
            csv_content2.append(S[1])
            
    # write res into csv file
    with open('a1_3.4.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerows(csv_content1)
        csvwriter.writerows([csv_content2])
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()
    
    # 3.1
    X_train, X_test, y_train, y_test, iBest = class31(args.input)
    
    # 3.2
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)
    
    # 3.3    
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    
    # 3.4
    class34(args.input, iBest)

    
    
