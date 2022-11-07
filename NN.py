import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import neural_network
from sklearn.model_selection import validation_curve
from learning_curve import single_valid, plot_learning_curve
from sklearn.model_selection import ShuffleSplit
from numpy.random import seed
import tensorflow as tf
import warnings
import time

warnings.filterwarnings('ignore')

#Source: https://www.kaggle.com/code/hhllcks/neural-net-with-gridsearch/notebook
#https://www.kaggle.com/code/jamesleslie/titanic-neural-network-for-beginners/notebook
#https://www.kaggle.com/code/carlmcbrideellis/very-simple-neural-network-for-classification/notebook
def nn_test(X,y, output, title):
    # X = df.drop([output], axis=1)
    # y = df[output]
       


    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


    # cols = X_train.columns
    
    # scaler = StandardScaler()

    # X_train = scaler.fit_transform(X_train)

    # X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)   
    # X_train = pd.DataFrame(X_train, columns=[cols])
    # X_test = pd.DataFrame(X_test, columns=[cols])
    clf = neural_network.MLPClassifier()
    if title == 'Stroke Prediction':
        X_train_over = scaler.fit_transform(X_train_over)
        X_test_over = scaler.transform(X_test_over)
        X_train_over = pd.DataFrame(X_train_over, columns=[cols])
        X_test_over = pd.DataFrame(X_test_over, columns=[cols])
        clf.fit(X_train_over,y_train_over)
        y_pred = clf.predict(X_test_over)
        y_train_pred = clf.predict(X_train_over)
        print(title+' Model balanced accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_test_over, y_pred)))
        print(title+' Training-set accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_train_over, y_train_pred))) 
        print(title+' Training set score with criterion entropy: {:.4f}'.format(clf.score(X_train_over, y_train_over)))
        print(title+' Test set score with criterion entropy: {:.4f}'.format(clf.score(X_test_over, y_test_over)))
    else:
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        y_train_pred = clf.predict(X_train)        
        print(title+' Model balanced accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_test, y_pred)))
        print(title+' Training-set accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_train, y_train_pred))) 
        print(title+' Training set score with criterion entropy: {:.4f}'.format(clf.score(X_train, y_train)))
        print(title+' Test set score with criterion entropy: {:.4f}'.format(clf.score(X_test, y_test)))    
    
    parameters = {'solver': ['adam'], 'max_iter': [500], 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':[(10,10), (10,10,10), (10,10,10,10)]}
    
    # parameters = {'solver': ['lbfgs','adam'], 'max_iter': [500,1000,1500], 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(5, 12), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
    clf_grid = GridSearchCV(neural_network.MLPClassifier(), parameters, n_jobs=-1)
    clf_grid.fit(X,y)
    
    print("-----------------Original Features--------------------")
    print("Best score: %0.4f" % clf_grid.best_score_)
    print("Using the following parameters:")
    print(clf_grid.best_params_)
    
    
    estimator = clf_grid.best_estimator_   
        
    start_train = time.time()
    estimator.fit(X_train,y_train)
    end_train = time.time()
    train_time = end_train-start_train
    
    start_train_predict = time.time()
    y_train_pred = estimator.predict(X_train)
    end_train_predict = time.time()
    train_predict_time = end_train_predict-start_train_predict  
    
    start_test_predict = time.time()
    y_test_pred = estimator.predict(X_test)
    end_test_predict = time.time()
    test_predict_time = end_test_predict-start_test_predict
    
    print("\n*******************************************************\n")
    
    print(title+' Timing for Training: {:.6f}'. format(train_time))
    print(title+' Timing for Train Predict: {:.6f}'. format(train_predict_time))
    print(title+' Timing for Test Predict: {:.6f}'. format(test_predict_time))

    print("\n*******************************************************\n")
        
    print(title+' GridSearch Model balanced accuracy score: {0:0.4f}'. format(balanced_accuracy_score(y_test, y_test_pred)))
    print(title+' GridSearch Training-set accuracy score: {0:0.4f}'. format(balanced_accuracy_score(y_train, y_train_pred))) 
    print(title+' GridSearch Training set score: {:.4f}'.format(estimator.score(X_train, y_train)))
    print(title+' GridSearch Test set score: {:.4f}'.format(estimator.score(X_test, y_test)))
    cv = ShuffleSplit(n_splits=50, test_size=0.3, random_state=0)
    title1 = "Balanced Accuracy Learning Curves (ANN) for " + title
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    plot_learning_curve(
        estimator,
        title1,
        X,
        y,
        axes=axes,
        ylim=(0.5, 1.01),

        n_jobs=4,
        scoring="accuracy",
    )

    plt.savefig('./plots/'+title+' ann_learning_balanced_accuracy_curve_.png')  
    plt.clf()
    

    
    param_range = range(5,25)
    train_scores, test_scores = validation_curve(
        neural_network.MLPClassifier(solver = 'adam', max_iter=500),
        X,
        y,
        param_name="hidden_layer_sizes",
        param_range=param_range,
        scoring="balanced_accuracy",
        n_jobs=2,
    )
        
    single_valid(title+"Hidden Layer Size Validation (adam)", train_scores, test_scores, param_range, "ANN hidden layers (adam)")   
    
    param_range = 10.0 ** -np.arange(1, 7)
    train_scores, test_scores = validation_curve(
        neural_network.MLPClassifier(solver = 'adam', max_iter=500),
        X,
        y,
        param_name="alpha",
        param_range=param_range,
        scoring="balanced_accuracy",
        n_jobs=2,
    )
        
    single_valid(title+"Alpha Validation (adam)", train_scores, test_scores, param_range, "ANN alpha (adam)")   
    # param_range = range(5,20)
    # train_scores, test_scores = validation_curve(
    #     neural_network.MLPClassifier(solver = 'lbfgs', max_iter=500),
    #     X,
    #     y,
    #     param_name="hidden_layer_sizes",
    #     param_range=param_range,
    #     scoring="balanced_accuracy",
    #     n_jobs=2,
    # )
        
    # single_valid(title+"Hidden Layer Size Validation (lbfgs)", train_scores, test_scores, param_range, "ANN hidden layers (lbfgs)")   

    return
        







if __name__ == "__main__":
    
    data = "./data/heart.csv"
    df = pd.read_csv(data)
    X = df.drop(["output"], axis=1)

    y = df["output"]
    nn_test(X,y, "output", "Heart Failure Prediction")
