import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models import LinearRegression
from models import LogisticRegression
from models import KNearestNeighbors

scaler = StandardScaler()                                # <------------------- remove standard scaler once testing against my scaler.


# Import Data Sets
print("Importing Data.")
housing_df = pd.read_csv('C:/Development/source/CalebFurley/UReCA/python-testing/datasets/housing-data.csv')
diabetes_df = pd.read_csv('C:/Development/source/CalebFurley/UReCA/python-testing/datasets/diabetes-data.csv')

# Preprocess Housing Data
print("Preprocessing Housing Data.")
housing_df = housing_df.replace({'yes':1.0, 'no':0.0, 'furnished':1.0, 'semi-furnished':0.5, 'unfurnished':0.0})
housing_train,housing_test = np.split(housing_df.sample(frac=1),[400],axis=0)
housing_train_x, housing_train_y = housing_train.iloc[0:, 1:].values, housing_train['price'].values
housing_test_x, housing_test_y = housing_test.iloc[0:, 1:].values, housing_test['price'].values
housing_train_x = scaler.fit_transform(housing_train_x)
housing_test_x = scaler.transform(housing_test_x)

# Preprocess Diabetes Data
print("Preprocessing Diabetes Data.")
diabetes_train_set, diabetes_test_set = np.split(diabetes_df.sample(frac=1),[450],axis=0)
diabetes_train_x, diabetes_train_y = diabetes_train_set.iloc[0:,0:-2].values, diabetes_train_set.iloc[0:,-1].values
diabetes_test_x, diabetes_test_y = diabetes_test_set.iloc[0:,0:-2].values, diabetes_test_set.iloc[0:,-1].values
diabetes_train_x = (diabetes_train_x - np.mean(diabetes_train_x)) / np.std(diabetes_train_x)        # <--------------- Use this math to build
diabetes_test_x = (diabetes_test_x - np.mean(diabetes_test_x)) / np.std(diabetes_test_x)            # <--------------- scaler for tools module.

####################################################################################################################


# Model Testing
print("About to train...\n\n")

##Linear-Regression --------------- [Score confirmed accurate vs sk-learn model]
#print("Training Linear Regression Model.")
#model = LinearRegression()
#model.train(housing_train_x, housing_train_y, 0.01, 500)
#model.predict(housing_test_x)
##print("R^2 Score = ", score, "\n\n")
#el model

##Logistic-Regression ------------- [WIP]
#print("Training Logistic Regression Model.")
#model = LogisticRegression()
#model.train(diabetes_train_x, diabetes_train_y, 0.01, 500)    # <----- training here [WIP]
#model.predict(diabetes_test_x)
#score = model.score(diabetes_test_x, diabetes_test_y)
#print("R^2 Score = ", score, "\n\n")
#del model


# K Nearest Testing here.

model = KNearestNeighbors(5, 2) # first int is 'k', second is number of classes to vote on.

model.train(diabetes_train_x, diabetes_train_y)
model.predict(diabetes_test_x)
score = model.score(diabetes_test_x, diabetes_test_y)
print("Score[KNN] = ", score, "\n\n")
del model
