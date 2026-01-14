import pandas as pd
import numpy as np
from collections import Counter
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import f1_score
from imblearn.over_sampling import SVMSMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

TFIDF_PATH = "../../dataset/features/"
CFG2VEC_PATH = "../../dataset/features/cfg2vec/"
LABEL_PATH = "../../dataset/label_6166.csv"


def run_fusion_model(sequence_feature_path, structure_feature_path):
    print("-" * 80)
    print(f"Running fusion experiment:")
    print(f"  Sequence Feature: {sequence_feature_path.split('/')[-1]}")
    print(f"  Structure Feature: {structure_feature_path.split('/')[-1]}")
    print("-" * 80)

    labelset = pd.read_csv(LABEL_PATH, index_col="address")
    sequence_features = pd.read_csv(sequence_feature_path, index_col="address")
    structure_features = pd.read_csv(structure_feature_path, index_col="address")

    combined_features = sequence_features.join(structure_features, lsuffix='_seq', rsuffix='_struct')
    datatable = combined_features.join(labelset)
    datatable.dropna(inplace=True)

    labels = datatable["label"]
    features = datatable.drop(columns=['label'])

    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)

    features_df = pd.DataFrame(features_imputed, index=features.index, columns=features.columns)

    seq_data = features_df.iloc[:, :len(sequence_features.columns)].to_numpy()
    struct_data = features_df.iloc[:, len(sequence_features.columns):].to_numpy()

    scaler = MinMaxScaler()
    scaled_struct_data = scaler.fit_transform(struct_data)

    norm_struct_data_list = []
    for vec in scaled_struct_data:
        norm = np.linalg.norm(vec)
        norm_struct_data_list.append(vec / norm if norm > 0 else vec)
    norm_struct_data = np.array(norm_struct_data_list)

    data = np.hstack((seq_data, norm_struct_data)).astype(np.float32)
    target = labels.to_numpy()

    print(f"Final Data Shape: {data.shape}, Label Distribution: {Counter(target)}")

    pipeline = Pipeline([
        ('smote', SVMSMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'))
    ])

    param_grid = {
        'rf__n_estimators': range(100, 300, 10),
        'rf__max_depth': range(10, 30, 1)
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(data, target)

    print("\n Best parameters found by GridSearchCV:", grid_search.best_params_)
    print(f" Best F1-score from GridSearchCV: {grid_search.best_score_:.4f}")

    best_estimator = grid_search.best_estimator_

    print("\n--- Final Model Performance (Mean +/- Std Dev) ---")
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for metric in scoring_metrics:
        scores = cross_val_score(best_estimator, data, target, cv=5, scoring=metric, n_jobs=-1)
        print(f"{metric.capitalize():<10}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print("\n")


def run_experiment_fixed_structure():
    print("=" * 25, "Starting Experiment 1: Fixed Structure Feature", "=" * 25)

    fixed_structure_feature = CFG2VEC_PATH + "filteredcfg_cfg2vec_5.csv"
    sequence_features_to_test = [
        TFIDF_PATH + "filteredcfg_1gram_tfidf.csv",
        TFIDF_PATH + "filteredcfg_2gram_tfidf.csv",
        TFIDF_PATH + "filteredcfg_3gram_tfidf.csv",
        TFIDF_PATH + "filteredcfg_4gram_tfidf.csv"
    ]

    for seq_feature in sequence_features_to_test:
        run_fusion_model(seq_feature, fixed_structure_feature)

    print("=" * 35, "Experiment 1 Finished", "=" * 35)


def run_experiment_fixed_sequence():
    print("=" * 25, "Starting Experiment 2: Fixed Sequence Feature", "=" * 25)

    fixed_sequence_feature = TFIDF_PATH + "filteredcfg_2gram_tfidf.csv"

    for i in range(1, 11):
        structure_feature_to_test = CFG2VEC_PATH + f"filteredcfg_cfg2vec_{i}.csv"
        run_fusion_model(fixed_sequence_feature, structure_feature_to_test)

    print("=" * 35, "Experiment 2 Finished", "=" * 35)


if __name__ == "__main__":
    run_experiment_fixed_structure()
    run_experiment_fixed_sequence()
    print("\nAll experiments completed successfully! 🎉")