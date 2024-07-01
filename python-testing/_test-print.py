import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models import LinearRegression

### IMPORT-DATA ###
print("Importing Data.")
df = pd.read_csv('C:/Users/caleb/Code/Trabajo/UReCA/python-testing/simple-housing-data.csv')
df.head()

### PREPROCESSING ###
print("Preprocessing Data.")
# replace strings with floats
df = df.replace({'yes':1.0, 'no':0.0, 'furnished':1.0, 'semi-furnished':0.5, 'unfurnished':0.0})
# Split data into train/test
train,test = np.split(df.sample(frac=1),[400],axis=0)
# Split data into x/y
train_x, train_y = train.iloc[0:, 1:].values, train['price'].values
test_x, test_y = test.iloc[0:, 1:].values, test['price'].values
# Scale Data
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x) 
# Newlines to console
print("About to train.\n\n\n")

##########################################################################
### MODEL-TESTING ###
#########################################################################

#Linear-Regression --------------- [Score confirmed accurate vs sk-learn model]
print("Training Linear Regression Model.")
model = LinearRegression()
model.train(train_x, train_y, 0.01, 500)
model.predict(test_x)
score = model.score(test_x, test_y)
print("R^2 Score = ", score)
