import pandas as pd
import numpy as np
import os
from sklearn.datasets import make_classification

# configuration
n_samples = 10000
n_features = 10
n_informative = 2
n_redundant = 2
n_useless = n_features - n_informative - n_redundant

important_indices = list(range(n_informative + n_redundant))
unimportant_indices = list(range(n_informative + n_redundant, n_features))

drift_point = n_samples // 2
drift_magnitude = 3.0

output_dir = "data"
file_1_data_name = os.path.join(output_dir, "drift_in_important_features_data.csv")
file_1_class_name = os.path.join(output_dir, "drift_in_important_features_class.csv")
file_2_data_name = os.path.join(output_dir, "drift_in_unimportant_features_data.csv")
file_2_class_name = os.path.join(output_dir, "drift_in_unimportant_features_class.csv")

def generate_asymmetric_drift_dataset(
        data_file_name: str,
        class_file_name: str,
        n_samples: int,
        drift_point: int,
        drift_feature_indices: list,
        apply_concept_drift: bool = False,
        concept_drift_feature_index: int = None
):
    print(f"--- generating dataset for: {os.path.basename(data_file_name)} ---")

    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        n_clusters_per_class=1,
        flip_y=0.05,
        random_state=41
    )

    x_post_drift = x[drift_point:].copy()
    y_post_drift = y[drift_point:].copy()

    print(f"applying feature drift to columns: {drift_feature_indices}")
    for feature_index in drift_feature_indices:
        x_post_drift[:, feature_index] += drift_magnitude

    if apply_concept_drift:
        if concept_drift_feature_index is None:
            raise ValueError("concept_drift_feature_index must be specified if apply_concept_drift is True.")
        print(f"applying concept drift based on column index: {concept_drift_feature_index}")

        median_val = np.median(x_post_drift[:, concept_drift_feature_index])
        drift_indices = x_post_drift[:, concept_drift_feature_index] > median_val
        y_post_drift[drift_indices] = 1 - y_post_drift[drift_indices]

    x_final = np.vstack((x[:drift_point], x_post_drift))
    y_final = np.hstack((y[:drift_point], y_post_drift))

    df_data = pd.DataFrame(x_final)
    df_data.to_csv(data_file_name, index=False, header=False)

    df_class = pd.DataFrame(y_final)
    df_class.to_csv(class_file_name, index=False, header=False)

    print(f"successfully generated files:")
    print(f"  data: '{data_file_name}'")
    print(f"  class: '{class_file_name}'")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("this script will generate two datasets for drift detection experiments.")
    print("each dataset will be split into a 'data' file and a 'class' file.")
    print(f"important features are considered to be indices: {important_indices}")
    print(f"unimportant features are considered to be indices: {unimportant_indices}\n")

    generate_asymmetric_drift_dataset(
        data_file_name=file_1_data_name,
        class_file_name=file_1_class_name,
        n_samples=n_samples,
        drift_point=drift_point,
        drift_feature_indices=important_indices,
        apply_concept_drift=True,
        concept_drift_feature_index=important_indices[0]
    )

    generate_asymmetric_drift_dataset(
        data_file_name=file_2_data_name,
        class_file_name=file_2_class_name,
        n_samples=n_samples,
        drift_point=drift_point,
        drift_feature_indices=unimportant_indices,
        apply_concept_drift=False
    )