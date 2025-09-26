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

tfpath = "../../dataset/features/"
tfidfpath = "../../dataset/features/"
cfg2vecpath = "../../dataset/features/cfg2vec/"
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


def train_tf():
    print("filteredcfg_tf")
    labelset = pd.read_csv(labelpath, index_col="address")

    # 1gram tf
    print("1gram tf start")
    train_smote_rf(labelset, tfpath + "filteredcfg_1gram_tf.csv")
    # 2gram tf
    print("2gram tf start")
    train_smote_rf(labelset, tfpath + "filteredcfg_2gram_tf.csv")
    # 3gram tf
    print("3gram tf start")
    train_smote_rf(labelset, tfpath + "filteredcfg_3gram_tf.csv")
    # 4gram tf
    print("4gram tf start")
    train_smote_rf(labelset, tfpath + "filteredcfg_4gram_tf.csv")
    print("train ngram tf end")


def train_tfidf():
    print("filteredcfg_tfidf")
    labelset = pd.read_csv(labelpath, index_col="address")

    # 1gram tfidf
    print("1gram tfidf start")
    train_smote_rf(labelset, tfidfpath + "filteredcfg_1gram_tfidf.csv")
    # 2gram tfidf
    print("2gram tfidf start")
    train_smote_rf(labelset, tfidfpath + "filteredcfg_2gram_tfidf.csv")
    # 3gram tfidf
    print("3gram tfidf start")
    train_smote_rf(labelset, tfidfpath + "filteredcfg_3gram_tfidf.csv")
    # 4gram tfidf
    print("4gram tfidf start")
    train_smote_rf(labelset, tfidfpath + "filteredcfg_4gram_tfidf.csv")
    print("train ngram tfidf end")


def train_cfg2vec():
    print("filteredcfg_cfg2vec")
    labelset = pd.read_csv(labelpath, index_col="address")

    print("cfg2vec_1")
    train_smote_rf(labelset, cfg2vecpath + "filteredcfg_cfg2vec_1.csv")
    # 2
    print("cfg2vec_2")
    train_smote_rf(labelset, cfg2vecpath + "filteredcfg_cfg2vec_2.csv")
    # 3
    print("cfg2vec_3")
    train_smote_rf(labelset, cfg2vecpath + "filteredcfg_cfg2vec_3.csv")
    # 4
    print("cfg2vec_4")
    train_smote_rf(labelset, cfg2vecpath + "filteredcfg_cfg2vec_4.csv")
    # 5
    print("cfg2vec_5")
    train_smote_rf(labelset, cfg2vecpath + "filteredcfg_cfg2vec_5.csv")
    # 6
    print("cfg2vec_6")
    train_smote_rf(labelset, cfg2vecpath + "filteredcfg_cfg2vec_6.csv")
    # 7
    print("cfg2vec_7")
    train_smote_rf(labelset, cfg2vecpath + "filteredcfg_cfg2vec_7.csv")
    # 8
    print("cfg2vec_8")
    train_smote_rf(labelset, cfg2vecpath + "filteredcfg_cfg2vec_8.csv")
    # 9
    print("cfg2vec_9")
    train_smote_rf(labelset, cfg2vecpath + "filteredcfg_cfg2vec_9.csv")
    # # 10
    print("cfg2vec_10")
    train_smote_rf(labelset, cfg2vecpath + "filteredcfg_cfg2vec_10.csv")

    print("train cfg2vec end")


def ronghe(labelset, featurepath):
    labelset = pd.read_csv(labelpath, index_col="address")
    featurepath1 = featurepath
    featuredata1 = pd.read_csv(featurepath1, index_col="address")

    featurepath2 = cfg2vecpath + "frequency_ratio.csv"
    featuredata2 = pd.read_csv(featurepath2, index_col="address")

    featuredata3 = featuredata1.join(featuredata2)
    datatable = featuredata3.join(labelset)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_imputed = imputer.fit_transform(datatable)

    # 然后继续归一化处理
    featvec = data_imputed[:, :featuredata2.shape[1]]
    m = MinMaxScaler()
    m.fit(featvec)
    v = m.transform(featvec)

    nn = []
    for i in v:
        magnitude = np.linalg.norm(i)
        if magnitude == 0:
            normalized_vector = i
        else:
            normalized_vector = i / magnitude
        nn.append(normalized_vector)
    nv = np.array(nn)

    data1 = nv
    data2 = datatable[featuredata1.columns].to_numpy()
    data = np.hstack((data1, data2))
    print(data.shape)

    label = datatable["label"]
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

    estimator = RandomForestClassifier(n_estimators=paras[index][0],
                                       max_depth=paras[index][1],
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


def train_ronghe():
    print("ronghe")
    labelset = pd.read_csv(labelpath, index_col="address")

    # 1gram tf
    print("1gram tf start")
    ronghe(labelset, tfpath + "filteredcfg_1gram_tf.csv")
    # 2gram tf
    print("2gram tf start")
    ronghe(labelset, tfpath + "filteredcfg_2gram_tf.csv")
    # 3gram tf
    print("3gram tf start")
    ronghe(labelset, tfpath + "filteredcfg_3gram_tf.csv")
    # 4gram tf
    print("4gram tf start")
    ronghe(labelset, tfpath + "filteredcfg_4gram_tf.csv")

    # 1gram tfidf
    print("1gram tfidf start")
    ronghe(labelset, tfidfpath + "filteredcfg_1gram_tfidf.csv")
    # 2gram tfidf
    print("2gram tfidf start")
    ronghe(labelset, tfidfpath + "filteredcfg_2gram_tfidf.csv")
    # 3gram tfidf
    print("3gram tfidf start")
    ronghe(labelset, tfidfpath + "filteredcfg_3gram_tfidf.csv")
    # 4gram tfidf
    print("4gram tfidf start")
    ronghe(labelset, tfidfpath + "filteredcfg_4gram_tfidf.csv")
    print("train ngram tfidf end")


if __name__ == "__main__":
    train_tf()
    train_tfidf()
    train_cfg2vec()
    train_ronghe()
