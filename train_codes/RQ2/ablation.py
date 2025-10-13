import os
import warnings
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SVMSMOTE
from imblearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

labelpath = "../../dataset/label_6166.csv"
tfidfpath = "../../dataset/features/"
cfg2vecpath = "../../dataset/features/cfg2vec/"
ablation_subdir_path = "../../dataset/features/ablation/"


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


def train_noSmote(labelset, path1, path2):
    print(f"\n--- Ablation: -Oversampling ---")

    featuredata1 = pd.read_csv(path1, index_col="address")
    featuredata2 = pd.read_csv(path2, index_col="address")

    combined_features = featuredata1.join(featuredata2, lsuffix='_1', rsuffix='_2')
    datatable = combined_features.join(labelset)
    datatable.dropna(inplace=True)

    features = datatable.drop(columns=['label'])
    target = datatable["label"].to_numpy()
    data = features.to_numpy()

    print("Feature shape:", data.shape)
    print("Class distribution:", Counter(target))

    estimator = RandomForestClassifier(n_estimators=100,
                                       max_depth=10,
                                       random_state=42,
                                       n_jobs=-1)

    evaluate_model_cv(estimator, data, target, use_smote=False)


def train_smote_rf(labelset, featurepath, variant_name):
    print(f"\n--- Ablation: {variant_name} ---")
    featuredata = pd.read_csv(featurepath, index_col="address")
    datatable = featuredata.join(labelset)
    datatable.dropna(inplace=True)

    features = datatable.drop(columns=['label'])
    target = datatable["label"].to_numpy()
    data = features.to_numpy()

    print("Feature shape:", data.shape)
    print("Class distribution:", Counter(target))

    estimator = RandomForestClassifier(n_estimators=160,
                                       max_depth=10,
                                       random_state=42,
                                       n_jobs=-1)

    evaluate_model_cv(estimator, data, target, use_smote=True)


def train_smote_rf_combined(labelset, path1, path2, variant_name):
    print(f"\n--- Ablation: {variant_name} ---")
    print(f"  (+) Base: {os.path.basename(path2)}")
    print(f"  (+) Variant: {os.path.basename(path1)}")

    featuredata1 = pd.read_csv(path1, index_col="address")
    featuredata2 = pd.read_csv(path2, index_col="address")

    combined_features = featuredata1.join(featuredata2, lsuffix='_1', rsuffix='_2')
    datatable = combined_features.join(labelset)
    datatable.dropna(inplace=True)

    features = datatable.drop(columns=['label'])
    target = datatable["label"].to_numpy()
    data = features.to_numpy()

    print("Combined feature shape:", data.shape)
    print("Class distribution:", Counter(target))

    estimator = RandomForestClassifier(n_estimators=160,
                                       max_depth=10,
                                       random_state=42,
                                       n_jobs=-1)

    evaluate_model_cv(estimator, data, target, use_smote=True)


if __name__ == "__main__":
    label_set = pd.read_csv(labelpath, index_col="address")

    print("\n" + "=" * 25 + " Starting Ablation Experiments " + "=" * 25)

    # –Oversampling: without training vector oversampling.
    train_noSmote(label_set,
                  os.path.join(tfidfpath, "filteredcfg_2gram_tfidf.csv"),
                  os.path.join(cfg2vecpath, "filteredcfg_cfg2vec_4.csv"))

    # –Structural: training without structural features.
    train_smote_rf(label_set,
                   os.path.join(tfidfpath, "filteredcfg_2gram_tfidf.csv"),
                   variant_name="-Structural")

    # –Semantic: training without sematic features.
    train_smote_rf(label_set,
                   os.path.join(cfg2vecpath, "filteredcfg_cfg2vec_5.csv"),
                   variant_name="-Semantic")

    # –Filter: without CFG filtering.
    train_smote_rf_combined(label_set,
                            os.path.join(tfidfpath, "rawcfg_2gram_tfidf.csv"),
                            os.path.join(cfg2vecpath, "filteredcfg_cfg2vec_5.csv"),
                            variant_name="-Filter")

    # Fixed base feature for the next 3 experiments
    fixed_tfidf = os.path.join(tfidfpath, "filteredcfg_2gram_tfidf.csv")

    # –Merging: without node merging.
    train_smote_rf_combined(label_set,
                            os.path.join(ablation_subdir_path, "xiaorong_cfg2vec_nomerge_5.csv"),
                            fixed_tfidf,
                            variant_name="-Merging")

    # –Path: without integrating graph paths.
    train_smote_rf_combined(label_set,
                            os.path.join(ablation_subdir_path, "xiaorong_cfg2vec_nopath_5.csv"),
                            fixed_tfidf,
                            variant_name="-Path")

    # –Subgraph: without integrating subgraphs.
    train_smote_rf_combined(label_set,
                            os.path.join(ablation_subdir_path, "xiaorong_cfg2vec_nosubgraph_5.csv"),
                            fixed_tfidf,
                            variant_name="-Subgraph")

    print("\n" + "=" * 28 + " All Experiments Completed " + "=" * 28)