# -*- coding: utf-8 -*-
# USAGE: python kernelsvm.py forest_data.npz
# First argument is a full path to dataset forest_data.npz

import numpy as np
np.random.seed(7)
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import math, sys

def plotScore(score_all):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()             
    legends = [r'C=$10^{-3}$', r'C=$10^{-2}$', r'C=$10^{10}$']
    x_axis = range(len(score_all))

    for i in range(len(score_all)):
        ax.plot(score_all[i])
    for i in range(len(score_all)):
        ax.plot(score_all[i], 'ko') # dot on line
    ax.set_ylabel('Accuracy Score')
    ax.legend(legends, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Kernal SVM - Validation Accuracy with 3-fold cross validation')
    ax.set_xlabel('Sigma')
    ax.set_xticks(x_axis)
    ax.set_xticklabels([r'$10^{0}$',r'$10^{2}$',r'$10^{4}$'])
    
def main(argv=None):
    plot = ""
    if argv is None:
        data = sys.argv[1]
        if (len(sys.argv) > 2):
            plot = sys.argv[2] # "plot" to show plot
            
    with np.load(data) as forest_data:
        X_train = forest_data['data_training']
        Y_train = forest_data['label_training']
        X_valid = forest_data['data_val']
        Y_valid = forest_data['label_val']

    # normalize feature vector - only first 10 variables are quantitative variables
    X_train_cont = X_train[:, 0:10]
    X_train_cat = X_train[:, 10:]
    X_train_cont_normalized = preprocessing.scale(X_train_cont, axis=0)
    X_train_normalized = np.concatenate((X_train_cont_normalized, X_train_cat), axis=1)
    
    X_valid_cont = X_valid[:, 0:10]
    X_valid_cat = X_valid[:, 10:]
    X_valid_cont_normalized = preprocessing.scale(X_valid_cont, axis=0)
    X_valid_normalized = np.concatenate((X_valid_cont_normalized, X_valid_cat), axis=1)
  
    # kernel svm
    C = np.array([0.001, 0.01, math.pow(10,10)])
    sigma = np.array([1, 100, 10000])
    gamma = 1.0/(2.0*sigma*sigma)
    model = svm.SVC(kernel='rbf')
    grid = GridSearchCV(estimator=model, scoring='accuracy', \
                        param_grid=dict(C=C, gamma=gamma), cv=None, n_jobs=-1)
    grid.fit(X_train_normalized, Y_train)
    
    # Print results
    best_score = grid.score(X_valid_normalized, Y_valid)
    print("KERNAL SVM\nBest test score: %s" % best_score)
    
    print("Best Validation Accuracy for each parameter set:")
    score_iter1 = grid.cv_results_['split0_test_score']
    score_iter2 = grid.cv_results_['split1_test_score']
    score_iter3 = grid.cv_results_['split2_test_score']
    score_all = np.concatenate((score_iter1, score_iter2, score_iter3), axis=0) \
                .reshape(3,len(C)*len(sigma))
    score_all = np.max(score_all, axis=0)
    score_all = score_all.reshape(len(C),len(sigma))
    for i in range(len(C)):
        for j in range(len(sigma)):
            print("- C = %s, sigma = %s: Accuracy = %s" %(C[i], sigma[j], score_all[i][j]))
    
    # Show plot if plot == "plot"
    if (plot == "plot"):
        plotScore(score_all)
        
if __name__ == "__main__":
    main()
