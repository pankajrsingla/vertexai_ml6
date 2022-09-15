# 0. Load required modules:
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Imputation Transformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler # Encoding Transformers
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
import pickle
import numpy as np
from google.cloud import storage

# 1. Dataset preparation
df = pd.read_csv("https://raw.githubusercontent.com/pankajrsingla/vertexai_ml6/main/data/titanic.csv")

FEATURES = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
TARGET = 'Survived'
train_df, test_df = train_test_split(df, train_size=0.7,
                                     shuffle=True, random_state=42)
X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

# 2. Data preprocessing
numerical_features = ['Age', 'Fare', 'Pclass']
categorical_features = ['Sex', 'Embarked']

# 2.1 Creating a preprocessing pipeline

# 2.1.1 Defining the categorical pipeline
#   steps of the pipeline have the form: ('name', TransformerObject)
# handle_unknown in OrdinalEncoder is required for Vertex AI predictions.
categorical_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value', unknown_value = np.nan))
])

# 2.1.2 Defining the numerical pipeline
numerical_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('normalizer', StandardScaler())
])

# 2.1.3 Combining the two pipilines
preprocessing_pipe = ColumnTransformer(
    transformers=[
      # ('name', Transformer, [column names])
        ('num', numerical_pipe, numerical_features),
        ('cat', categorical_pipe, categorical_features)]
)

# 2.1.4 Adding the LightGBM classifier to the pipeline
full_pipe = Pipeline(steps=[
    ('preprocessor', preprocessing_pipe),
    ('clf', LGBMClassifier(n_estimators=20, random_state=42))
])

# 3. Train the model using full_pipe
model = full_pipe.fit(X_train, y_train)

print(f'Train accuracy: {model.score(X_train, y_train):.2f}')
print(f'Test accuracy: {model.score(X_test, y_test):.2f}')

# 4. For hyperparameter tuning and explainability, refer to
# https://github.com/ml6team/quick-tips/blob/main/structured_data/2021_02_26_scikit_learn_pipelines/scikit_learn_pipelines_and_lightgbm_titanic_dataset.ipynb

# 5. Save the model to GCP bucket:
