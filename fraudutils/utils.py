import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from sklearn.metrics import recall_score, precision_score, accuracy_score, make_scorer, confusion_matrix, average_precision_score, roc_auc_score
from pprint import pprint

def load_train_test_val_dataset(path, files=['train_df.csv', 'test_df.csv', 'val_df.csv']):
    data = []

    for file_name in files:
        df = pd.read_csv(path + file_name)
        data.append(df)
    return data

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          show_matrix=False,
                          show_desc=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if show_desc:
            print("Normalized confusion matrix")
    else:
        if show_desc:
            print('Confusion matrix, without normalization')

    if show_matrix:
        print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def classify(X_train, X_test, y_train, y_test, classifier, random_state=0, normalized=True):
    
    clf = classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auprc = average_precision_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred)
    
    print("Mean accuracy: {}".format(accuracy))
    print("Mean precision: {}".format(precision))
    print("Mean recall: {}".format(recall))
    print("AUPRC: {}".format(auprc))
    print("AUROC: {}".format(auroc))
    
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    plot_confusion_matrix(cm=cm, classes=['Not fraud', 'Fraud'], normalize=normalized)
    
    return {'accuracy': accuracy, 
            'precision': precision,
            'recall': recall,
            'AUPRC': auprc,
            'AUROC': auroc}

def stratified_crossvalidation(classifier, X, y, cv, scoring, normalized=False):
    from sklearn.model_selection import StratifiedKFold
    from collections import defaultdict
    from statistics import mean
    
    X = np.array(X)
    y = np.array(y)
    
    sfk = StratifiedKFold(cv)
    results = defaultdict(list)
    
    general_y_pred = []
    general_y_real = []

    for train_index, test_index in sfk.split(X, y):
        
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        clf = classifier
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        general_y_pred.extend(y_pred)
        general_y_real.extend(y_test)
        
        for score_name, score_function in scoring.items():
            results[score_name].append(score_function(clf, X_test, y_test))
    
    for score_name, score_results in results.items():
        results[score_name] = mean(score_results)
    
    cm = confusion_matrix(y_true=general_y_real, y_pred=general_y_pred)
    plot_confusion_matrix(cm=cm, classes=['Not fraud', 'Fraud'], normalize=normalized)

    results = dict(results)
    pprint(results)

    return results
