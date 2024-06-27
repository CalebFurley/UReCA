import numpy as np

from models import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.array([[2, 1], [2, 2], [3, 2], [4, 3]])

reg = LinearRegression(X, y, 1, 500);
reg.train();
reg.predict(np.array([[3, 5]]))
score = reg.score(X, y);
print(score)