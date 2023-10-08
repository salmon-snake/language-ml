# Necessary imports
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve

# Plots the training and validation scores for classifier type learning
def train_curves_plot(clf, x, y,title=''):
    train_sizes, train_scores, test_scores = learning_curve(clf,x,y,train_sizes=[i / 10 for i in range(1, 9)])
    avg_trainscores = np.mean(train_scores,axis=1)
    avg_testscores = np.mean(test_scores,axis=1)
    plt.title(title)
    plt.plot(train_sizes, avg_trainscores, color='r', label='Training scores')
    plt.plot(train_sizes, avg_testscores, color='g', label='Validation scores')
    plt.xlabel('Training set size')
    plt.ylabel('Classification accuracy')
    plt.legend(loc='best')
    plt.show(block=False)
    plt.figure()

# Plots the training vs testing accuracy for each label
def train_test_accuracy_plot(train_acc, test_acc, labels, title=''):
    size = len(labels)
    barWidth = 0.25
    fig = plt.subplots(figsize=(12,8))
    br1 = np.arange(size)
    br2 = [x + barWidth for x in br1]
    plt.bar(br1, train_acc, color='r',width=barWidth,label='Training')
    plt.bar(br2, test_acc, color='b',width=barWidth,label='Testing')
    plt.xticks([x + barWidth/2 for x in range(size)], labels)
    plt.title(title)
    plt.ylim((0.8,1))
    plt.legend()
    plt.show(block=False)
    plt.figure()
def train_test_by_treesize(x_train, x_test, y_train, y_test,label):
    training = []
    testing = []
    it = range(8,30,2)
    for i in it:
        tree = DecisionTreeClassifier(max_depth=i)
        tree.fit(x_train,y_train)
        y_train_pred = tree.predict(x_train)
        y_test_pred = tree.predict(x_test)
        training.append(accuracy_score(y_train,y_train_pred))
        testing.append(accuracy_score(y_test,y_test_pred))
    plt.plot(it,training, color='r', label='Training scores')
    plt.plot(it,testing, color='g', label='Validation scores')
    plt.legend()
    plt.title("Decision tree training and validation accuracies by tree size, " + label)
    plt.show(block=False)
    plt.figure()

# Organizes and runs all tests    
def run_tests(path,labels,clf):
    df = pd.read_csv(path)
    print(df)
    # For testing, comment out for final results
    #df = df.sample(frac=0.5)
    # Forming the feature vector
    x = df.drop(columns=labels)
    # We will append the accuracy of each classification here
    train_scores = []
    test_scores = []
    # Run tests one label (language) at a time
    for label in labels:
        y = df[label]
        # First plot, measuring classifier accuracy as a function of training set size
        # Computational.ly intensive, can be commented out
        # train_curves_plot(clf,x,y,title=str(clf)+' testing and validation accuracy by training size, '+label)
        # Splitting data into training and testing sets based on first plot results
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        # Another optional plot for decision trees
        train_test_by_treesize(x_train,x_test,y_train,y_test,label)
        # Fitting the model and grabbing the training and testing accuracies
        clf.fit(x_train, y_train)
        y_train_pred = clf.predict(x_train)
        y_test_pred = clf.predict(x_test)
        y_train_accuracy = accuracy_score(y_train,y_train_pred)
        y_test_accuracy = accuracy_score(y_test,y_test_pred)
        # Adding the train score to our list
        train_scores.append(y_train_accuracy)
        test_scores.append(y_test_accuracy)
    train_test_accuracy_plot(train_scores,test_scores,labels,title=str(clf)+' training and testing accuracy by language')
    
# Names of the CSV column headers that we use for labels
labels = ['Finnish','French','English','Latin','Spanish']
# Classifiers used
regr = LogisticRegression()
tree = DecisionTreeClassifier(max_depth=20)

# Filepaths for the full feature vector and the trimmed version
path_full = 'features_full.csv'
path_trimmed = 'features.csv'

# Activates the run function
run_tests(path_full,labels,tree)











    
    


