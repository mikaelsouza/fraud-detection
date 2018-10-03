
# coding: utf-8

# In[2]:


import keras
import numpy as np
from sklearn import model_selection, preprocessing, metrics, utils
import pandas as pd
import matplotlib.pyplot as plt
#import fraudutils as futils
rs = 1


# In[31]:


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


# ### Loading Dataset

# In[32]:


train, test, val = load_train_test_val_dataset('../../../data/processed/give-me-some-credit/')

X_train = train.drop('SeriousDlqin2yrs', axis=1)
y_train = train['SeriousDlqin2yrs']

X_test = test.drop('SeriousDlqin2yrs', axis=1)
y_test = test['SeriousDlqin2yrs']

X_val = val.drop('SeriousDlqin2yrs', axis=1)
y_val = val['SeriousDlqin2yrs']


# In[33]:


y_train = np.array([[x[0], x[1], 0] for x in keras.utils.to_categorical(y_train)])
y_test = np.array([[x[0], x[1], 0] for x in keras.utils.to_categorical(y_test)])
y_val = np.array([[x[0], x[1], 0] for x in keras.utils.to_categorical(y_val)])


# ### Defining sizes

# In[34]:


# Input for generator network
noise_size = 100
label_size = 3
representation_size = 10
hidden_layer = 128


# ### Using Paper's Hyperparameters

# In[35]:


adam = keras.optimizers.Adam(lr=0.00001)


# ### Defining Generator Model

# In[36]:


gen_input_size = noise_size
# Generating input layer with functional approach
gen_input = keras.layers.Input(shape=(gen_input_size,))
# Dropbout for fun
gen_drop_layer = keras.layers.Dropout(0.1)(gen_input)
# Generating hidden layer using functional approach
gen_hidden = keras.layers.Dense(hidden_layer, activation='tanh')(gen_drop_layer)
# Generating output layer using functional approach
gen_output = keras.layers.Dense(representation_size, activation='sigmoid')(gen_hidden)
# Defining model
gen_model = keras.models.Model(inputs=gen_input, outputs=gen_output)
gen_model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])


# ### Defining Discriminator Model

# In[37]:


dis_input_size = representation_size 
# Discriminator input layer
dis_input = keras.layers.Input(shape=(representation_size,))
# Discriminator hidden layer
dis_drop_layer = keras.layers.Dropout(0.1)(dis_input)
dis_hidden = keras.layers.Dense(hidden_layer, activation='tanh')(dis_input)
# Discriminator output layer
dis_output = keras.layers.Dense(3, activation='softmax')(dis_hidden)
# Defining model
dis_model = keras.models.Model(inputs=dis_input, outputs=dis_output)
dis_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


# In[38]:


# Combined Network
dis_model.trainable = False
gan_input = keras.layers.Input(shape=(noise_size,))

x = gen_model(gan_input)
gan_output = dis_model(x)

gan = keras.models.Model(inputs=gan_input, outputs=gan_output)
gan.compile(loss='categorical_crossentropy', optimizer=adam)


# In[39]:


generator_losses = []
discriminator_losses = []

def train(epochs=1, batch_size=128):
    batch_count = int(X_train.shape[0] /  batch_size)
    print("Epochs: ", epochs)
    print("Batch size: ", batch_size)
    print("Batches per Epoch: ", batch_count)
    
    for e in range(1, epochs + 1):
        for _ in range(batch_count):
            noise = np.random.normal(0, 1, size=[batch_size, noise_size])
            indexes = np.random.randint(0, X_train.shape[0], size=batch_size)
            real_batch = np.array(X_train)[indexes]
            generated_batch = gen_model.predict(noise)
            
            X_joined = np.concatenate([real_batch, generated_batch])
            y_joined = np.concatenate([y_train[indexes], np.array([[0, 0, 0.9]] * batch_size)])

            dis_model.trainable = True
            dis_loss = dis_model.train_on_batch(X_joined, y_joined)
            
            noise = np.random.normal(0, 1, size=[batch_size, noise_size])
            y_gen = np.array([[0.05, 0.05, 0.9]] * batch_size)
            dis_model.trainable = False
            gen_loss = gan.train_on_batch(noise, y_gen)
        generator_losses.append(gen_loss)
        discriminator_losses.append(dis_loss)
        if e % 100 == 0:
            print('\n', '-' * 15, 'Epoch {}'.format(e), '-' * 15)
            prediction = dis_model.predict(X_test).argmax(1).reshape(-1, 1)
            auc_roc_score = metrics.roc_auc_score(y_test, prediction)
            auc_pr_score = metrics.average_precision_score(y_test, prediction)
            print("ROC: {}\nPR: {}".format(auc_roc_score, auc_pr_score))


# In[40]:


train(10000)

