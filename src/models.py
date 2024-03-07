import pandas as pd
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sys.path.append('src')


df = pd.read_csv('data/liver_disease_preprocessed.csv')  

x = df.drop("Class", axis = 1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.20, random_state= 23)

def logisticRegression(x_train_param, x_test_param, y_train_param, y_test_param):
    clf = LogisticRegression(max_iter= 5000)
    clf.fit(x_train_param, y_train_param)

    y_pred = clf.predict(x_test_param)

    acc = accuracy_score(y_test_param, y_pred)
    precision = precision_score(y_test_param, y_pred)
    recall = recall_score(y_test_param, y_pred)
    f1 = f1_score(y_test_param, y_pred)
    print("Logistic Regression model accuracy (in %): ", acc*100)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("f1 score: ", f1)
    return [acc, precision, recall, f1]

def knn_model(x_train_param, x_test_param, y_train_param, y_test_param):
    knn = KNeighborsClassifier(n_neighbors=75)

    knn.fit(x_train_param, y_train_param)

    y_pred= knn.predict(x_test_param)
    acc = accuracy_score(y_test_param, y_pred)
    precision = precision_score(y_test_param, y_pred)
    recall = recall_score(y_test_param, y_pred)
    f1 = f1_score(y_test_param, y_pred)
    print("Logistic Regression model accuracy (in %): ", acc*100)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("f1 score: ", f1)
    return [acc, precision, recall, f1]

def naiveBayes(x_train_param, x_test_param, y_train_param, y_test_param):
    gnb = GaussianNB()
    gnb.fit(x_train_param, y_train_param)

    y_pred = gnb.predict(x_test_param)
    acc = accuracy_score(y_test_param, y_pred)
    precision = precision_score(y_test_param, y_pred)
    recall = recall_score(y_test_param, y_pred)
    f1 = f1_score(y_test_param, y_pred)
    print("Logistic Regression model accuracy (in %): ", acc*100)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("f1 score: ", f1)
    return [acc, precision, recall, f1]