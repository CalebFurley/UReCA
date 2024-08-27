import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

from toolset import Scaler

current_directory = os.getcwd()

housing_df = pd.read_csv(f"{current_directory}/test-enviroment/datasets/housing-data.csv")
diabetes_df = pd.read_csv(f"{current_directory}/test-enviroment/datasets/diabetes-data.csv")

housing_df = housing_df.replace({'yes':1.0, 'no':0.0, 'furnished':1.0, 'semi-furnished':0.5, 'unfurnished':0.0})
diabetes_df = diabetes_df.replace({'tested_positive':1.0, 'tested_negative':0.0})

sklearn_housing_df = housing_df.copy()
sklearn_diabetes_df = diabetes_df.copy()
ureca_housing_df = housing_df.copy()
ureca_diabetes_df = diabetes_df.copy()

sklearn_scaler = StandardScaler()
sklearn_housing_np = sklearn_scaler.fit_transform(sklearn_housing_df)
sklearn_diabetes_np = sklearn_scaler.fit_transform(sklearn_diabetes_df)
sklearn_housing_df = pd.DataFrame(sklearn_housing_np, columns=sklearn_housing_df.columns)
sklearn_diabetes_df = pd.DataFrame(sklearn_diabetes_np, columns=sklearn_diabetes_df.columns)

ureca_scaler = Scaler()
ureca_scaler.scale(ureca_housing_df)
ureca_scaler.scale(ureca_diabetes_df)
# TESTING URECA SCALER HERE ##########################################

print("Ureca Housing Scaled\n", ureca_housing_df.head(), "\n")
print("Sklearn Housing Scaled\n", sklearn_housing_df.head(), "\n")
print("Ureca Diabetes Scaled\n", ureca_diabetes_df.head(), "\n")
print("Sklearn Diabetes Scaled\n", sklearn_diabetes_df.head(), "\n")