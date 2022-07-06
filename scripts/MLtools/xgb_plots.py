
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier



def xgbplot(model, trainX, trainY, testX, testY):

	    ## plot the logloss and error figures
    eval_set = [(trainX, trainY), (testX, testY)]
    model.fit(trainX, trainY, early_stopping_rounds=10, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)
    results = model.evals_result()
    # make predictions for test data
    # y_pred = model.predict(testX)
    # predictions = [round(value) for value in y_pred]
    # # evaluate predictions
    # accuracy = accuracy_score(testY, predictions)

	    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['error'])
    print(epochs)
    x_axis = range(0, epochs)
    
    plt.style.use("ggplot")
    plt.rcParams.update({'font.size': 18}) # Increase font size

    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss', loc='top')
    plt.xlabel('Epoch', loc='right')
    plt.title('XGBoost Log Loss')
    plt.savefig('Figures/XGBLogLoss_defult_dPhiCor.pdf', bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()

    # plot classification error
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('Classification Error', loc='top')
    plt.xlabel('Epoch', loc='right')
    plt.title('XGBoost Classification Error')
    plt.savefig('Figures/XGBClassError_dPhiCor.pdf', bbox_inches='tight')
    plt.show()
    # predict the labels of the test set
    predictedY = model.predict(testX)

    print('\nTesting Confusion Matrix:\n')
    print(confusion_matrix(testY, predictedY))

    sn.heatmap(confusion_matrix(testY, predictedY))
    plt.figure() 

    return 