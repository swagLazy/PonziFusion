import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')

featurepath = "../dataset/features/"
cfg2vecpath = "../dataset/features/cfg2vec/"
labelpath = "../dataset/label_6166.csv"


def load_data(tfidf_file, cfg_num):
    print(f"Loading Data: {tfidf_file} + CFG-{cfg_num} ... ")
    try:
        labelset = pd.read_csv(labelpath, index_col="address")
        tfidf_df = pd.read_csv(featurepath + tfidf_file, index_col="address")
        cfg_df = pd.read_csv(cfg2vecpath + f"filteredcfg_cfg2vec_{cfg_num}.csv", index_col="address")

        data = tfidf_df.join(cfg_df, lsuffix='_tf', rsuffix='_cfg').join(labelset).dropna()

        y = data['label'].to_numpy()
        X = data.drop(columns=['label']).to_numpy()
        X = MinMaxScaler().fit_transform(X)

        return X, y
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def run_evaluation(X, y):
    seeds = [30707, 91551, 79826, 37359, 38665, 15684, 85891, 166, 99687, 12339]
    rf_model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features='sqrt',
        max_depth=20,
        criterion='entropy',
        class_weight='balanced_subsample',
        n_jobs=-1
    )

    print(f"--- Evaluating PonziFusion (3-gram + CFG-4) ---")

    metrics_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }

    for seed in seeds:
        rf_model.set_params(random_state=seed)
        pipeline = ImbPipeline([
            ('smote', SVMSMOTE(random_state=seed)),
            ('model', rf_model)
        ])

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        for metric in metrics_results.keys():
            score = cross_val_score(pipeline, X, y, cv=cv, scoring=metric, n_jobs=-1).mean()
            metrics_results[metric].append(score)

    for metric, scores in metrics_results.items():
        scores_arr = np.array(scores)
        print(f"{metric.capitalize():<10}: {scores_arr.mean():.4f} (+/- {scores_arr.std():.4f})")
        print(f"CV_Results_{metric}: {scores_arr.tolist()}")
        print("-" * 40)


def main():
    X, y = load_data("filteredcfg_3gram_tfidf.csv", 4)
    if X is not None:
        run_evaluation(X, y)


if __name__ == "__main__":
    main()