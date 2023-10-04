# Necessary imports
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve

# Filepath for csv
path = 'features.csv'
# Names of the CSV column headers
languages = ['Finnish','French','English','Latin','Spanish']
# Shuffles the import for train-test split, random state allows for controlled randomness
df = pd.read_csv(path)

# Forming the feature vector
x = df.drop(columns=languages+['Word'])
y = df[languages]

#
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)

# Anything computationally intensive is placed in a function
def train_curves_plot(clf, x, y):
    train_sizes, train_scores, test_scores = learning_curve(clf,x,y,train_sizes=[i / 10 for i in range(1, 10)])
    avg_trainscores = np.mean(train_scores,axis=1)
    avg_testscores = np.mean(test_scores,axis=1)
    plt.plot(train_sizes, avg_trainscores, color='r', label='Training scores')
    plt.plot(train_sizes, avg_testscores, color='g', label='Validation scores')
    plt.xlabel('Training set size')
    plt.ylabel('Classification accuracy')
    plt.legend(loc='best')
    plt.show()

def train_test_accuracy_plot(train_acc, test_acc, title, ylim=(0,1)):
    barWidth = 0.25
    fig = plt.subplots(figsize=(12,8))
    br1 = np.arange(len(languages))
    br2 = [x + barWidth for x in br1]
    plt.bar(br1, train_acc, color='r',width=barWidth,label='Training accuracy')
    plt.bar(br2, test_acc, color='b',width=barWidth,label='Testing accuracy')
    plt.xticks([x + barWidth/2 for x in range(len(languages))], languages)
    plt.title(title)
    plt.ylim(ylim)
    plt.legend()
    plt.show()
    
# Returns predicted labels for train and test given classifier plus train and test sets
def reg_pred(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    return y_train_pred, y_test_pred

clf = LogisticRegression(max_iter=100, n_jobs=8)
regr = LinearRegression(n_jobs=8)

train_curves_plot(clf,x,y['English'])

train_acc = []
test_acc = []
title = "Figure III - Linear regression training vs testing accuracy, by language"
for lang in languages:
    y_train_s = y_train[lang]
    y_test_s = y_test[lang]
    y_train_pred, y_test_pred = reg_pred(regr, x_train, y_train_s, x_test, y_test_s)
    y_train_accuracy = mean_squared_error(y_train_s,y_train_pred)
    y_test_accuracy = mean_squared_error(y_test_s,y_test_pred)
    train_acc.append(y_train_accuracy)
    test_acc.append(y_test_accuracy)
train_test_accuracy_plot(train_acc,test_acc,title)       

train_acc = []
test_acc = []
title = "Figure II - Logistic regression training vs testing accuracy, by language"
for lang in languages:
    y_train_s = y_train[lang]
    y_test_s = y_test[lang]
    y_train_pred, y_test_pred = reg_pred(clf, x_train, y_train_s, x_test, y_test_s)
    y_train_accuracy = accuracy_score(y_train_s,y_train_pred)
    y_test_accuracy = accuracy_score(y_test_s,y_test_pred)
    train_acc.append(y_train_accuracy)
    test_acc.append(y_test_accuracy)
train_test_accuracy_plot(train_acc,test_acc,title,ylim=(0.8,1))       









    
    


