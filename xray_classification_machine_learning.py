import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import glob
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
import cv2
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

seed = 2021

TRAIN_NORMAL_DATA_PATH = glob.glob("./data/chest_xray/train/NORMAL/*.jpeg")
TRAIN_PNEUMONIA_DATA_PATH = glob.glob("./data/chest_xray/train/PNEUMONIA/*.jpeg")

TEST_NORMAL_DATA_PATH = glob.glob("./data/chest_xray/test/NORMAL/*.jpeg")
TEST_PNEUMONIA_DATA_PATH = glob.glob("./data/chest_xray/test/PNEUMONIA/*.jpeg")

train_paths = TRAIN_NORMAL_DATA_PATH + TRAIN_PNEUMONIA_DATA_PATH
test_paths = TEST_NORMAL_DATA_PATH + TEST_PNEUMONIA_DATA_PATH

train_labels = [0] * len(TRAIN_NORMAL_DATA_PATH) + [1] * len(TRAIN_PNEUMONIA_DATA_PATH)
test_targets = [0] * len(TEST_NORMAL_DATA_PATH) + [1] * len(TEST_PNEUMONIA_DATA_PATH)

train_data = list()
test_data = list()


def compute_confusion_matrix(y_true, y_predictions):
    cm = confusion_matrix(y_true, y_predictions)
    plt.figure()
    plot_confusion_matrix(cm, figsize=(14, 10), cmap=plt.cm.Blues)
    plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=20)
    plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=20)
    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.show()


for path in train_paths:
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (50, 50))
    train_data.append(np.array(img, dtype=float).flatten())

for path in test_paths:
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (50, 50))
    test_data.append(np.array(img, dtype=float).flatten())

X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_labels, test_size=0.2, shuffle=True)

models = list()
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=seed)))

# variables to hold the results and names
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

    model.fit(X_train, y_train)
    predicted = model.predict(test_data)
    print(f"Classification report for classifier {name}:\n"
          f"{metrics.classification_report(test_targets, predicted)}\n")

    compute_confusion_matrix(test_targets,predicted)
    print("#############################################################")
