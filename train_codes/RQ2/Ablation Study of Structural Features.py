import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, roc_auc_score, \
    accuracy_score
import pandas as pd
import numpy as np
from imblearn.over_sampling import SVMSMOTE
from collections import Counter
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

datapath = "../../dataset/"
ablationpath = "../../dataset/features/"
labelpath = "../../dataset/label_6166.csv"


def train_smote_rf(labelset, featurepath):
    print(featurepath)
    featuredata = pd.read_csv(featurepath, index_col="address")
    datatable = featuredata.join(labelset)
    feat_data = datatable[featuredata.columns]
    label = datatable["label"]
    data = feat_data.to_numpy()
    target = label.to_numpy()
    print(data.shape, Counter(target))

    x_train, x_test, y_train, y_test = train_test_split(data,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=42)
    smote1 = SVMSMOTE(random_state=42)
    x_res, y_res = smote1.fit_resample(x_train, y_train)

    paras = []
    cross = []
    r = range(100, 300, 10)
    f = range(10, 30, 1)

    for i in r:
        for j in f:
            estimator = RandomForestClassifier(n_estimators=i, n_jobs=-1, random_state=42, max_depth=j)
            estimator.fit(x_res, y_res)
            test_pred = estimator.predict(x_test)
            score = roc_auc_score(y_test, test_pred)
            paras.append((i, j))
            cross.append(score)
    index = cross.index(max(cross))
    print(cross)
    print(paras)
    print("best paras:", paras[index], cross[index])

    estimator = RandomForestClassifier(n_estimators=160,
                                       max_depth=10,
                                       random_state=42,
                                       n_jobs=-1)

    estimator.fit(x_res, y_res)

    test_pred = estimator.predict(x_test)
    # test
    print("Accuracy:", accuracy_score(y_test, test_pred))
    print("precision:", precision_score(y_test, test_pred))
    print("recall:", recall_score(y_test, test_pred))
    print("F1:", f1_score(y_test, test_pred))
    print("roc_auc:", roc_auc_score(y_test, test_pred))


if __name__ == "__main__":

    label_set = pd.read_csv(labelpath, index_col="address")

    feature_names = [
        "rawcfg_1gram_tfidf.csv",
        "removedcfg_1gram_tfidf.csv",
        "filteredcfg_1gram_tfidf.csv"
    ]

    for fn in feature_names:
        feature_file = os.path.join(ablationpath, fn)
        train_smote_rf(label_set, feature_file)
