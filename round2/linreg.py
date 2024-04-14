import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## To use statsmodels for linear regression
import statsmodels.formula.api as smf

## To use sklearn for linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

data0 = pd.read_csv("./IMC testing/round2/round-2-island-data-bottle/prices_round_2_day_0.csv", delimiter=";")
dataneg1 = pd.read_csv("./IMC testing/round2/round-2-island-data-bottle/prices_round_2_day_-1.csv", delimiter=";")
data1 = pd.read_csv("./IMC testing/round2/round-2-island-data-bottle/prices_round_2_day_1.csv", delimiter=";")

datacombined = pd.concat([dataneg1, data0, data1], axis = 0)
datacombined.reset_index(drop=True, inplace=True)

def lm(data):
    data.drop('EXPORT_TARIFF', axis = 1, inplace = True)
    data.drop('IMPORT_TARIFF', axis = 1, inplace = True)
    data.drop(['TRANSPORT_FEES', "DAY"], axis = 1, inplace = True)

    data["ORCHIDS_-1"] = data["ORCHIDS"].shift(1)
    data["ORCHIDS_-2"] = data["ORCHIDS"].shift(2)
    data["ORCHIDS_-3"] = data["ORCHIDS"].shift(3)

    data.dropna(inplace=True)

    # X = data[['ORCHIDS_-3', 'ORCHIDS_-1', 'ORCHIDS_-2', 'SUNLIGHT', 'HUMIDITY']]
    X = data[['ORCHIDS_-3', 'ORCHIDS_-1', 'ORCHIDS_-2']]
    y = data['ORCHIDS']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    # Instantiate and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # Print the coefficients and intercept
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

lm(datacombined)

print(datacombined)