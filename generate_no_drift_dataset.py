import numpy as np
import pandas as pd

# Parameters
n_instances = 1000
n_features = 8
random_seed = 42  # for reproducibility

# Set random seed
np.random.seed(random_seed)

# --- 1. Generate Features (Stable P(X)) ---
# We'll create a mix of feature types/distributions.
# The parameters of these distributions are fixed for all instances.

features = pd.DataFrame()

# Feature 1 & 2: Standard Normal distribution
features['F1'] = np.random.normal(loc=0, scale=1, size=n_instances)
features['F2'] = np.random.normal(loc=5, scale=2, size=n_instances)

# Feature 3 & 4: Uniform distribution
features['F3'] = np.random.uniform(low=-2, high=2, size=n_instances)
features['F4'] = np.random.uniform(low=0, high=10, size=n_instances)

# Feature 5: Normal distribution with different mean/std
features['F5'] = np.random.normal(loc=-3, scale=0.5, size=n_instances)

# Feature 6: A discrete feature (e.g., categories 0, 1, 2)
features['F6'] = np.random.randint(low=0, high=3, size=n_instances) # Values will be 0, 1, or 2

# Feature 7: Another Normal distribution
features['F7'] = np.random.normal(loc=1, scale=1, size=n_instances)

# Feature 8: Correlated with F1 slightly, plus some noise
# To make it interesting, let's make F8 somewhat dependent on F1 but still random.
# This relationship is part of the stable P(X) as it's defined once.
noise_for_f8 = np.random.normal(loc=0, scale=0.8, size=n_instances)
features['F8'] = 0.6 * features['F1'] + noise_for_f8 # F8 has a linear relationship with F1 + noise

# --- 2. Define a Stable Relationship for Target Class (P(Y|X)) ---
# We'll create a linear combination of some features, pass it through a sigmoid
# to get a probability, and then assign classes based on this probability.
# The weights and the rule itself are FIXED.

# Define fixed weights for the linear combination
# These weights determine how important each feature is and its direction of influence.
# Not all features need to be used, or they can have different impacts.
w = {
    'F1': 0.5,
    'F2': -0.2,
    'F3': 1.0,
    'F4': 0.1,
    'F5': -0.8,
    'F6': 0.4, # Note: F6 is discrete, this weight will apply to its numerical value
    'F7': 0.0, # F7 will have no direct impact on the class in this rule
    'F8': 1.2
}
intercept = -0.5 # A fixed intercept

# Calculate the linear combination (logit)
logit = intercept
for i in range(1, n_features + 1):
    feature_name = f'F{i}'
    logit += w[feature_name] * features[feature_name]

# Apply sigmoid function to get probabilities
# probability = 1 / (1 + e^(-logit))
probabilities_class1 = 1 / (1 + np.exp(-logit))

# --- 3. Assign Target Class (Y) based on probabilities ---
# This introduces controlled randomness based on P(Y|X), making it not strictly deterministic
# but still following a stable rule.
target = np.random.binomial(1, probabilities_class1, size=n_instances)

# Combine features and target into a single DataFrame
dataset = features.copy()
dataset['target'] = target

# --- Display some information about the generated dataset ---
print("Generated Dataset Head:")
print(dataset.head())
print("\nDataset Info:")
dataset.info()
print("\nTarget Class Distribution:")
print(dataset['target'].value_counts(normalize=True))
print(f"\nShape of the dataset: {dataset.shape}")

dataset.drop(['target'], axis=1).to_csv('data.csv', index=False)
dataset['target'].to_csv('class.csv', index=False)