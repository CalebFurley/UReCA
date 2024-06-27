import numpy as np
import pandas as pd
from models import LinearRegression
from sklearn.preprocessing import StandardScaler

#############################################
############## Import data ##################
#############################################
print("Importing Data.")
df = pd.read_csv('C:/Users/caleb/Code/Trabajo/UReCA/python-testing/simple-housing-data.csv')
df.head()
#############################################


#############################################
############## Preprocess data ##############
#############################################
print("Preprocessing Data.")

# replace all strings with floats
df = df.replace({'yes':1.0, 'no':0.0, 'furnished':1.0, 'semi-furnished':0.5, 'unfurnished':0.0})

# split up data set into training and testing
train,test = np.split(df.sample(frac=1),[400],axis=0)

# split training and testing into x and y (use .values)
train_x, train_y = train.iloc[0:, 1:].values, train['price'].values
test_x, test_y = test.iloc[0:, 1:].values, test['price'].values

# Scale the features (train_x shows equation, test_x uses Scaler from sklearn)
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)  # Note: not fit_transform
    
#############################################


#############################################
############## Train model ##################
#############################################
print("About to train.\n\n\n")

# Create a Linear Regression model
model = LinearRegression()
model.train(train_x, train_y, 0.01, 500)
model.predict(test_x)
score = model.score(test_x, test_y)
print("R^2 Score = ", score)
