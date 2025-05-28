import numpy as np
import pandas as pd

# parameters
n_instances = 10000
n_features = 8
random_seed = 42

# set random seed
np.random.seed(random_seed)

# generate features (stable p(x))
features = pd.DataFrame()

features['F1'] = np.random.normal(loc=0, scale=1, size=n_instances)
features['F2'] = np.random.normal(loc=5, scale=2, size=n_instances)
features['F3'] = np.random.uniform(low=-2, high=2, size=n_instances)
features['F4'] = np.random.uniform(low=0, high=10, size=n_instances)
features['F5'] = np.random.normal(loc=-3, scale=0.5, size=n_instances)
features['F6'] = np.random.randint(low=0, high=3, size=n_instances)
features['F7'] = np.random.normal(loc=1, scale=1, size=n_instances)

# f8 is correlated with f1 plus some noise
noise_for_f8 = np.random.normal(loc=0, scale=0.8, size=n_instances)
features['F8'] = 0.6 * features['F1'] + noise_for_f8

# define a stable relationship for target class (p(y|x))
w = {
    'F1': 0.5,
    'F2': -0.2,
    'F3': 1.0,
    'F4': 0.1,
    'F5': -0.8,
    'F6': 0.4,
    'F7': 0.0,
    'F8': 1.2,
}
intercept = -0.5

logit = intercept
for i in range(1, n_features + 1):
    feature_name = f'F{i}'
    logit += w[feature_name] * features[feature_name]

probabilities_class1 = 1 / (1 + np.exp(-logit))

# assign target class (y) based on probabilities
target = np.random.binomial(1, probabilities_class1, size=n_instances)

# combine features and target into a single dataframe
dataset = features.copy()
dataset['target'] = target

print("Generated Dataset Head:")
print(dataset.head())
print("\nDataset Info:")
dataset.info()
print("\nTarget Class Distribution:")
print(dataset['target'].value_counts(normalize=True))
print(f"\nShape of the dataset: {dataset.shape}")

dataset.drop(['target'], axis=1).to_csv('data.csv', index=False)
dataset['target'].to_csv('class.csv', index=False)