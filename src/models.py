import pandas as pd
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

sys.path.append('src')

df = pd.read_csv('/data/liver_disease_preprocessed.csv')  

x = df.drop("Class", axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=23)

def logisticRegression(x_train_param, x_test_param, y_train_param, y_test_param):
    clf = LogisticRegression(max_iter=5000)
    clf.fit(x_train_param, y_train_param)

    y_pred = clf.predict(x_test_param)

    acc = accuracy_score(y_test_param, y_pred)
    precision = precision_score(y_test_param, y_pred)
    recall = recall_score(y_test_param, y_pred)
    f1 = f1_score(y_test_param, y_pred)
    print("Logistic Regression model accuracy (in %): ", acc * 100)
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
    print("KNN model accuracy (in %): ", acc*100)
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
    print("Naive Bayes model accuracy (in %): ", acc*100)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("f1 score: ", f1)
    return [acc, precision, recall, f1]

def svm_model(x_train_param, x_test_param, y_train_param, y_test_param):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(x_train_param)
    X_test_scaled = scaler.transform(x_test_param)

    param_grid = {
        'C': [0.1, 1, 10, 100], 
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],  
        'class_weight': [None, 'balanced'] 
    }

    svm_classifier = SVC()

    grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_param)

    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    best_svm_model = grid_search.best_estimator_

    y_pred = best_svm_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test_param, y_pred)
    print("SVM model accuracy (in %):", accuracy*100)
    print(classification_report(y_test_param, y_pred))

    precision = precision_score(y_test_param, y_pred)
    recall = recall_score(y_test_param, y_pred)
    f1 = f1_score(y_test_param, y_pred)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("f1 score: ", f1)

print("Logistic Regression results\n")
logisticRegression(x_train, x_test, y_train, y_test)
print("\nKNN results\n")
knn_model(x_train, x_test, y_train, y_test)
print("\nNaive Bayes results\n")
naiveBayes(x_train, x_test, y_train, y_test)
print("\nSVM results\n")
svm_model(x_train, x_test, y_train, y_test)
