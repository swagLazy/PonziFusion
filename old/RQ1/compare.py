from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
from imblearn.over_sampling import SVMSMOTE
from imblearn.pipeline import Pipeline
from collections import Counter
import json
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

featurepath = "../../dataset/features/"
cfg2vecpath = "../../dataset/features/cfg2vec/"
labelpath = "../../dataset/label_6166.csv"


def loadfeature(filename):
    return pd.read_csv(featurepath + filename, index_col="address")


def get_full_data(featurename):
    print(f"\nLoading and combining TF-IDF ({featurename}) and CFG2Vec features...")

    labelset = pd.read_csv(labelpath, index_col="address")
    featuredata1 = loadfeature(featurename)
    featuredata2 = pd.read_csv(cfg2vecpath + "filteredcfg_cfg2vec_5.csv", index_col="address")

    featuredata3 = featuredata1.join(featuredata2, lsuffix='_tfidf', rsuffix='_cfg2vec')
    datatable = featuredata3.join(labelset)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    labels = datatable['label']
    features = datatable.drop(columns=['label'])
    features_imputed = imputer.fit_transform(features)

    datatable_imputed = pd.DataFrame(features_imputed, index=features.index, columns=features.columns)
    datatable_imputed = datatable_imputed.join(labels)
    datatable_imputed.dropna(subset=['label'], inplace=True)

    target = datatable_imputed["label"].to_numpy()

    cfg2vec_features = datatable_imputed.iloc[:, featuredata1.shape[1]:featuredata1.shape[1]+featuredata2.shape[1]].to_numpy()

    tfidf_features = datatable_imputed.iloc[:, :featuredata1.shape[1]].to_numpy()

    scaler = MinMaxScaler()
    scaled_cfg2vec = scaler.fit_transform(cfg2vec_features)

    normalized_cfg2vec_list = []
    for i in scaled_cfg2vec:
        magnitude = np.linalg.norm(i)
        if magnitude == 0:
            normalized_vector = i
        else:
            normalized_vector = i / magnitude
        normalized_cfg2vec_list.append(normalized_vector)
    normalized_cfg2vec = np.array(normalized_cfg2vec_list)

    data = np.hstack((tfidf_features, normalized_cfg2vec))
    data = data.astype(np.float32)

    print(f"Final combined data shape: {data.shape}, Label distribution: {Counter(target)}")
    return data, target


def evaluate_model_cv(model, data, target, use_smote=False):
    if use_smote:
        pipeline = Pipeline([
            ('smote', SVMSMOTE(random_state=42)),
            ('model', model)
        ])
        model_name = type(model).__name__ + " (with SMOTE)"
    else:
        pipeline = model
        model_name = type(model).__name__

    print(f"--- Evaluating {model_name} ---")

    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for metric in scoring_metrics:
        try:
            scores = cross_val_score(pipeline, data, target, cv=5, scoring=metric, n_jobs=-1)
            print(f"{metric.capitalize():<10}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        except Exception as e:
            print(f"Could not calculate {metric}: {e}")
    print("-" * 40)


def train(featurename):
    data, target = get_full_data(featurename)

    rf_model = RandomForestClassifier(random_state=42)
    evaluate_model_cv(rf_model, data, target, use_smote=False)

    cat_model = CatBoostClassifier(random_state=42, verbose=0)
    evaluate_model_cv(cat_model, data, target, use_smote=False)

    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    evaluate_model_cv(xgb_model, data, target, use_smote=False)

    lgbm_model = LGBMClassifier(random_state=42)
    evaluate_model_cv(lgbm_model, data, target, use_smote=False)


def train1():
    print("===== Starting Combined Features Training =====")
    feature_files = [
        "filteredcfg_2gram_tfidf.csv"
    ]
    for file in feature_files:
        train(file)


if __name__ == "__main__":
    train1()