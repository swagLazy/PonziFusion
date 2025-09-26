from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, roc_auc_score, \
    accuracy_score
import pandas as pd
import numpy as np
from imblearn.over_sampling import SVMSMOTE
from collections import Counter
import json

featurepath = "../../dataset/features/"
midfeaturepath = "../../dataset/features/mid_features/"
labelpath = "../../dataset/label_6166.csv"


def loadfeature(filename):
    return pd.read_csv(featurepath + filename, index_col="address")


def loadmidfeature(filename):
    return pd.read_csv(midfeaturepath + filename, index_col="address")


def getdata(featurename):
    print(featurename)

    featuredata = loadfeature(featurename)
    labelset = pd.read_csv(labelpath, index_col="address")
    return split_data(featuredata, labelset)


def getvecdata(featurename):
    print(featurename)

    featuredata = loadfeature(featurename)
    labelset = pd.read_csv(labelpath, index_col="address")

    return split_vec(featuredata, labelset)


def split_vec(featuredata, labelset):
    datatable = featuredata.join(labelset)
    fdata = []
    for index in datatable.index:
        x = datatable.loc[index, "word2vec"]
        s = json.loads(x)
        fdata.append(s)
    data = np.array(fdata)
    label = datatable["label"]
    target = label.to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(data,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=42)

    return x_train, x_test, y_train, y_test


def split_data(featuredata, labelset):
    datatable = featuredata.join(labelset)
    feat_data = datatable[featuredata.columns]
    label = datatable["label"]
    data = feat_data.to_numpy()
    target = label.to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(data,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=42)

    return x_train, x_test, y_train, y_test


def train_model(model, x_train, x_test, y_train, y_test):
    estimator = model
    estimator.fit(x_train, y_train)
    test_pred = estimator.predict(x_test)
    print("accuracy:", accuracy_score(y_test, test_pred))
    print("precision:", precision_score(y_test, test_pred))
    print("recall:", recall_score(y_test, test_pred))
    print("F1:", f1_score(y_test, test_pred))
    print("roc_auc:", roc_auc_score(y_test, test_pred))


def train(featurename):
    print(featurename)
    x_train,x_test,y_train,y_test = getdata(featurename)
    # x_train, x_test, y_train, y_test = getvecdata(featurename)

    print("train RF:")
    model = RandomForestClassifier(random_state=42)
    train_model(model, x_train, x_test, y_train, y_test)
    print("\n")

    print("train  CAT:")
    model = CatBoostClassifier(random_state=42, logging_level="Silent")
    train_model(model, x_train, x_test, y_train, y_test)
    print("\n")

    print("train  XGB:")
    model = XGBClassifier(random_state=42)
    train_model(model, x_train, x_test, y_train, y_test)
    print("\n")

    print("train  LGBM:")
    model = LGBMClassifier(random_state=42)
    train_model(model, x_train, x_test, y_train, y_test)
    print("\n")


def train_tf():
    print("filteredcfg")
    train("filteredcfg_1gram_tf.csv")
    train("filteredcfg_2gram_tf.csv")
    train("filteredcfg_3gram_tf.csv")
    train("filteredcfg_4gram_tf.csv")

    print("rawcfg")
    train("rawcfg_1gram_tf.csv")
    train("rawcfg_2gram_tf.csv")
    train("rawcfg_3gram_tf.csv")
    train("rawcfg_4gram_tf.csv")

    print("removedcfg")
    train("removedcfg_1gram_tf.csv")
    train("removedcfg_2gram_tf.csv")
    train("removedcfg_3gram_tf.csv")
    train("removedcfg_4gram_tf.csv")


def train_tfidf():
    print("filteredcfg")
    train("filteredcfg_1gram_tfidf.csv")
    train("filteredcfg_2gram_tfidf.csv")
    train("filteredcfg_3gram_tfidf.csv")
    train("filteredcfg_4gram_tfidf.csv")

    print("rawcfg")
    train("rawcfg_1gram_tfidf.csv")
    train("rawcfg_2gram_tfidf.csv")
    train("rawcfg_3gram_tfidf.csv")
    train("rawcfg_4gram_tfidf.csv")

    print("removedcfg")
    train("removedcfg_1gram_tfidf.csv")
    train("removedcfg_2gram_tfidf.csv")
    train("removedcfg_3gram_tfidf.csv")
    train("removedcfg_4gram_tfidf.csv")



if __name__ == "__main__":
    train_tf()
    train_tfidf()
