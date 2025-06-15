import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

def generate_data(N=100):
    np.random.seed(42)
    data = []
    
    for _ in range(N):
        area = np.random.randint(50, 121)
        rooms = np.random.randint(1, 6)
        floor = np.random.randint(1, 11)
        year_of_construction = np.random.randint(1950, 2023)
        
        base_price = 100000
        area_factor = area * 2000
        room_factor = rooms * 15000
        floor_factor = floor * 5000 
        year_factor = (year_of_construction - 1950) * 1000
        random_factor = np.random.normal(0, 50000)
        
        price = base_price + area_factor + room_factor + floor_factor + year_factor + random_factor
        
        data.append([area, rooms, floor, year_of_construction, price])
    
    df = pd.DataFrame(data, columns=['area', 'rooms', 'floor', 'year_of_construction', 'price'])
    return df

def zad3():
    df = generate_data(100)
    print(df.head())

    X = df[['area', 'rooms', 'floor', 'year_of_construction']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")

    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Linear Regression: Actual vs Predicted Apartment Prices')
    plt.grid(True)
    plt.savefig(os.path.join('lab3_wykresy', 'apartment_price_prediction.png'))
    plt.close()

def zad4():
    months = np.array([i for i in range(1, 13)])
    temperatures = np.array([-2, 0, 5, 12, 18, 23, 26, 25, 21, 15, 8, 2])

    df = pd.DataFrame({
        'month': months,
        'temperature': temperatures
    })
    print(df)

    X_train, X_test, y_train, y_test = train_test_split(
        months.reshape(-1, 1), temperatures, test_size=2, random_state=42
    )

    max_degree = 4
    results = []

    plt.figure(figsize=(12, 6))
    colors = ['blue', 'green', 'red', 'cyan']

    plt.scatter(months, temperatures, color='black', label='Actual data')

    for degree in range(1, max_degree + 1):
        poly_model = make_pipeline(
            PolynomialFeatures(degree=degree),
            LinearRegression()
        )
        poly_model.fit(months.reshape(-1, 1), temperatures)
        
        X_range = np.arange(1, 13, 0.1).reshape(-1, 1)
        y_poly_pred = poly_model.predict(X_range)
        
        y_test_pred = poly_model.predict(X_test.reshape(-1, 1))
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        
        results.append({
            'degree': degree,
            'mse': mse,
            'r2': r2
        })
        
        plt.plot(X_range, y_poly_pred, color=colors[degree-1], 
                label=f'Degree {degree}, R²={r2:.4f}')

    plt.xlabel('Month')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature by Month: Polynomial Regression Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('lab3_wykresy', 'polynomial_regression.png'))
    plt.close()

    print("\nPolynomial Regression Results:")
    results_df = pd.DataFrame(results)
    print(results_df)

    best_degree = results_df.loc[results_df['r2'].idxmax()]['degree']
    print(f"\nBest polynomial degree based on R² score: {int(best_degree)}")

def load_data():
    df = pd.read_csv(os.path.join('lab3_dane', 'temperature_and_energy_consumption.csv'))
    print(df.head())
    return df

def zad5():
    df = load_data()
    X = df[['temperature']].values
    y = df['energy_consumption'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge())
    ])

    lasso_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ])

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    ridge_params = {'model__alpha': alphas}
    ridge_grid = GridSearchCV(ridge_pipeline, ridge_params, cv=5, scoring='r2')
    ridge_grid.fit(X_train, y_train)

    lasso_params = {'model__alpha': alphas}
    lasso_grid = GridSearchCV(lasso_pipeline, lasso_params, cv=5, scoring='r2')
    lasso_grid.fit(X_train, y_train)

    lr_pipeline.fit(X_train, y_train)

    best_ridge = ridge_grid.best_estimator_
    best_lasso = lasso_grid.best_estimator_

    y_pred_lr = lr_pipeline.predict(X_test)
    y_pred_ridge = best_ridge.predict(X_test)
    y_pred_lasso = best_lasso.predict(X_test)

    results = []
    for name, y_pred in [
        ('Linear Regression', y_pred_lr),
        ('Ridge', y_pred_ridge),
        ('Lasso', y_pred_lasso)
    ]:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({
            'Model': name,
            'MSE': mse,
            'R²': r2
        })

    print("\nModel Evaluation:")
    results_df = pd.DataFrame(results)
    print(results_df)

    print(f"\nBest Ridge alpha: {ridge_grid.best_params_['model__alpha']}")
    print(f"Best Lasso alpha: {lasso_grid.best_params_['model__alpha']}")

    plt.figure(figsize=(10, 6))
    test_idx = np.argsort(X_test.flatten())
    x_test_sorted = X_test[test_idx]
    y_test_sorted = y_test[test_idx]
    y_lr_sorted = y_pred_lr[test_idx]
    y_ridge_sorted = y_pred_ridge[test_idx]
    y_lasso_sorted = y_pred_lasso[test_idx]

    plt.scatter(X_test, y_test, alpha=0.5, label='Test Data')
    plt.plot(x_test_sorted, y_lr_sorted, 'r-', linewidth=2, label='Linear Regression')
    plt.plot(x_test_sorted, y_ridge_sorted, 'g-', linewidth=2, label='Ridge')
    plt.plot(x_test_sorted, y_lasso_sorted, 'b-', linewidth=2, label='Lasso')
    plt.xlabel('Temperature')
    plt.ylabel('Energy Consumption')
    plt.title('Temperature vs Energy Consumption: Model Predictions')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('lab3_wykresy', 'regression_comparison.png'))
    plt.close()

    lr_coef = lr_pipeline.named_steps['model'].coef_[0]
    ridge_coef = best_ridge.named_steps['model'].coef_[0]
    lasso_coef = best_lasso.named_steps['model'].coef_[0]

    print(f"Linear Regression: {lr_coef:.6f}")
    print(f"Ridge Regression: {ridge_coef:.6f}")
    print(f"Lasso Regression: {lasso_coef:.6f}")

    best_model_name = results_df.loc[results_df['R²'].idxmax()]['Model']
    print(f"\nBest model based on R² score: {best_model_name}")

def load_medical_data():
    df = pd.read_csv(os.path.join('lab3_dane', 'dane_medyczne.csv'))
    print(df.head())
    return df

def zad6():
    df = load_medical_data()

    X = df[['wiek', 'BMI', 'cisnienie_krwi', 'poziom_glukozy', 'cholesterol', 'kretynina']].values
    y = df['czas_przezycia'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    svr_linear_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(kernel='linear'))
    ])

    svr_rbf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(kernel='rbf'))
    ])

    svr_poly_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(kernel='poly'))
    ])

    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ])

    lasso_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=1.0))
    ])

    param_grid = {
        'model__C': [0.1, 1, 10, 100],
        'model__epsilon': [0.01, 0.1, 0.2, 0.5]
    }

    svr_linear_grid = GridSearchCV(svr_linear_pipeline, param_grid, cv=5, scoring='r2')
    svr_rbf_grid = GridSearchCV(svr_rbf_pipeline, param_grid, cv=5, scoring='r2')
    svr_poly_grid = GridSearchCV(svr_poly_pipeline, param_grid, cv=5, scoring='r2')

    models = {
        'SVR (Linear)': svr_linear_grid,
        'SVR (RBF)': svr_rbf_grid,
        'SVR (Polynomial)': svr_poly_grid,
        'Linear Regression': lr_pipeline,
        'Ridge': ridge_pipeline,
        'Lasso': lasso_pipeline
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    results = []
    predictions = {}

    for name, model in models.items():
        if isinstance(model, GridSearchCV):
            y_pred = model.best_estimator_.predict(X_test)
            best_params = model.best_params_
            print(f"{name} best parameters: {best_params}")
        else:
            y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MSE': mse,
            'R²': r2
        })
        
        predictions[name] = y_pred

    print("\nModel Evaluation:")
    results_df = pd.DataFrame(results)
    print(results_df)

    best_model_name = results_df.loc[results_df['R²'].idxmax()]['Model']
    print(f"\nBest model based on R² score: {best_model_name}")

    plt.figure(figsize=(10, 6))
    best_predictions = predictions[best_model_name]
    plt.scatter(y_test, best_predictions, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual Survival Time')
    plt.ylabel('Predicted Survival Time')
    plt.title(f'Actual vs Predicted: {best_model_name}')
    plt.grid(True)
    plt.savefig(os.path.join('lab3_wykresy', 'svr_model_results.png'))
    plt.close()

if __name__ == "__main__":
    zad3()
    zad4()
    zad5()
    zad6()