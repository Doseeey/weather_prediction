import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def make_prediction(criteria, time_span, debug_mode):
    df = pd.read_csv('Wroclaw.csv', sep=';', parse_dates=['date'])

    for column in ["temp_pow", "temp_grunt", "suma_opad_doba", "sr_pred_wiatr"]:
        df[column] = df[column].str.replace(",", ".").astype(float)

    df = df.dropna(subset=["temp_pow"])
    df = df.fillna(df.backfill())

    X = df[criteria]
    y = df["temp_pow"]

    X = X.fillna(X.backfill())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),  
        ('poly', PolynomialFeatures(degree=7, include_bias=False)),  
        ('ridge', Ridge())  
    ])

    param_grid = {
        'ridge__alpha': [0.2,0.3,0.35, 0.5,1, 10, 100, 1000],  
        'poly__degree': [1, 2, 2, 3,5]  
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    if debug_mode:
        print("Best parameters:", grid_search.best_params_)
        
    df["temp_pow_pred"] = best_model.predict(X)
    
    train_end_date = df['date'][len(X_train)-1]
    prediction_end_date = train_end_date + pd.Timedelta(days=time_span)

    df_filtered = df[(df['date'] > train_end_date) & (df['date'] <= prediction_end_date)]

    return df_filtered['temp_pow_pred']

    
columns = ["kier_wiatr", "temp_grunt", "suma_opad_doba", "sr_pred_wiatr", "wilg"]
time_span = 10
debug_mode = False

make_prediction(columns,time_span, debug_mode)