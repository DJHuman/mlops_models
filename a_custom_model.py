# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv("boston_housing.csv")

labels = df.pop('MEDV')

X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.01, random_state=42)

numeric_features = list(X_train.select_dtypes(include=np.number).columns.values)
excluded_features = ['Customer Id']
categorical_features = list(set(X_train.columns) - set(numeric_features) - set(excluded_features))

categorical_preprocessing = Pipeline(
    [
        ('Imputation', SimpleImputer(strategy='constant', fill_value='?', add_indicator=True)),
        ('One Hot Encoding', OneHotEncoder(handle_unknown='ignore')),
    ]
)

numeric_preprocessing = Pipeline(
    [('Imputation', SimpleImputer(strategy='constant', fill_value=-9999, add_indicator=True))]
)

preprocessing = make_column_transformer(
    (numeric_preprocessing, numeric_features),
    (categorical_preprocessing, categorical_features),
)

pipeline = Pipeline(
    [('Preprocessing', preprocessing),
     ('Random Forest', RandomForestRegressor(random_state = 1234, max_features=0.4,max_leaf_nodes=10,
                                              n_estimators = 100, bootstrap=False, min_samples_split=10,
                                              min_samples_leaf=5))]
)

pipeline.fit(X_train, y_train)

with open('custom_model.pkl', 'wb') as picklefile:
    pickle.dump(pipeline, picklefile)


# dataset = lgb.Dataset(df)

# dataset.save_binary('training.bin')

# validation = dataset.create_valid('validation.svm')

