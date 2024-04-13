import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

dataset = pd.read_csv('archive/insurance.csv')

import numpy as np
np.random.seed(42)

dataset["faixas_imc"] = pd.cut(dataset["imc"],
                               bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf],
                               labels=[1, 2, 3, 4, 5, 6])

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

dataset['genero_type'] = label_encoder.fit_transform(dataset['gênero'])
dataset['fumante_type'] = label_encoder.fit_transform(dataset['fumante'])
dataset['regiao_type'] = label_encoder.fit_transform(dataset['região'])

dataset_tratado = dataset.drop(columns = [ "imc", "fumante", "região", "gênero"]).copy()

X = dataset_tratado.drop(columns=['encargos'], axis=1)
y = dataset_tratado['encargos']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(y_test, predictions):
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse) 
    r2 = r2_score(y_test, predictions)
    errors = np.abs(y_test - predictions)
    relative_errors = errors / np.abs(y_test)
    mape = np.mean(relative_errors) * 100
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print('r²', r2)
    print(f"O MAPE é: {mape:.2f}%")

def run_model(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print()
    print(model)
    evaluate_model(y_test, predictions)

run_model(LinearRegression(), X_train, y_train, X_test)
run_model(DecisionTreeRegressor(), X_train, y_train, X_test)
run_model(RandomForestRegressor(), X_train, y_train, X_test)    
run_model(GradientBoostingRegressor(), X_train, y_train, X_test)