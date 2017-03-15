# -*- coding: utf-8 -*-
# USAGE: python randomforest.py forest_data.npz
# First argument is a full path to dataset forest_data.npz

import numpy as np
np.random.seed(7)
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import sys

def plotScore(oob_error, forest_size, sampling_rate, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()             
    legends = ['Forest Size = ' + str(i) for i in forest_size]

    for i in range(len(oob_error)):
        ax.plot(sampling_rate, oob_error[i])
    for i in range(len(oob_error)):
        ax.plot(sampling_rate, oob_error[i], 'ko') # dot on line
    ax.set_ylabel('Out-of-bag Error')
    ax.legend(legends, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title(title)
    ax.set_xlabel('Sampling Rate')
    
def main(argv=None):
    plot = ""
    if argv is None:
        data = sys.argv[1]
        if (len(sys.argv) > 2):
            plot = sys.argv[2] # "plot" to show plot
            
    with np.load(data) as forest_data:
        X_train = forest_data['data_training']
        Y_train = forest_data['label_training']

    # normalize feature vector - only first 10 variables are quantitative variables
    X_train_cont = X_train[:, 0:10]
    X_train_cat = X_train[:, 10:]
    X_train_cont_normalized = preprocessing.scale(X_train_cont, axis=0)
    X_train_normalized = np.concatenate((X_train_cont_normalized, X_train_cat), axis=1)
    
    ## random forest with datapoint sampling and max features
    forest_size = np.array([10, 20, 50])
    data_sampling = np.array([0.2, 0.4, 0.6, 0.8])
    
    oob_error1 = []
    for size in forest_size:
        for rate in data_sampling:
            model1 = RandomForestClassifier(n_estimators=size, max_depth=100, \
                    max_features=None, bootstrap=True, \
                    class_weight='balanced', oob_score=True, n_jobs=-1)
            model1.fit(X_train_normalized, Y_train, rate)
            oob_error1.append(1 - model1.oob_score_)
    
    
    ## random forest with feature sampling and max data point
    feature_sampling = np.array([0.2, 0.4, 0.6, 0.8])
    
    oob_error2 = []
    for size in forest_size:
        for rate in feature_sampling:
            model2 = RandomForestClassifier(n_estimators=size, max_depth=100, \
                    max_features=rate, bootstrap=True, \
                    class_weight='balanced', oob_score=True, n_jobs=-1)
            model2.fit(X_train_normalized, Y_train)
            oob_error2.append(1 - model2.oob_score_)
    
    ## Print results
    # Print result for random forest with data point sampling
    print("\nRANDOM FOREST WITH DATA POINT SAMPLING")
    print("Out-of-bag Error Estimates:")
    oob_error1 = np.array(oob_error1).reshape(len(forest_size), len(data_sampling))
    for i in range(len(forest_size)):
        for j in range(len(data_sampling)):
            print("- Forest Size = %i, data sampling rate = %s: Error = %s" \
                  %(forest_size[i], data_sampling[j], oob_error1[i][j]))
            
    # Plot for random forest with data point sampling
    if (plot == "plot"):
        plotScore(oob_error1, forest_size, data_sampling, \
                  'Random Forest with Data Point Sampling')
    
    # Print result for random forest with feature sampling
    print("\nRANDOM FOREST WITH FEATURE SAMPLING")
    print("Out-of-bag Error Estimates:")
    oob_error2 = np.array(oob_error2).reshape(len(forest_size), len(feature_sampling))
    for i in range(len(forest_size)):
        for j in range(len(feature_sampling)):
            print("- Forest Size = %i, feature sampling rate = %s: Error = %s" \
                  %(forest_size[i], feature_sampling[j], oob_error2[i][j]))
            
    # Plot for random forest with feature sampling
    if (plot == "plot"):
        plotScore(oob_error2, forest_size, feature_sampling, \
                  'Random Forest with Feature Sampling')
        
if __name__ == "__main__":
    main()
