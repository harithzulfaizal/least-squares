import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('usecons7687.csv')
varlst = [7, 6, 5, 4, 3, 2]

for var in varlst:
    X = X = df[['PURCHASE', 'GNP']].values.reshape(-1,2)
    #X = df.iloc[:,1:var].values # 12x6 matrix
    y = df['CONSUMER'].values # Column matrix

    x = np.ones((len(df),)) # Column matrix of ones
    A = np.column_stack((x, X)) # Vandermonde matrix

    coeff = np.linalg.inv(A.T@A)@A.T@y
    #print(coeff)

    ols = LinearRegression() # Sklearn's Ordinary Least Squares
    model = ols.fit(X, y)
    print(np.hstack((model.intercept_, model.coef_)))
