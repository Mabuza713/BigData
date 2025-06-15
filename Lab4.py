import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.interpolate import splrep, splev, CubicHermiteSpline, interp1d, CubicSpline
from scipy.signal import argrelextrema
from sklearn.metrics import mean_squared_error

road_traffic = pd.read_csv(r'C:\Users\kazwe\OneDrive\Pulpit\IT\STUDIA\Semestr4\Zarzycki L\LABY4\road_traffic.csv')
stocks_data = pd.read_csv(r'C:\Users\kazwe\OneDrive\Pulpit\IT\STUDIA\Semestr4\Zarzycki L\LABY4\stocks_data.csv')
energy_consumption = pd.read_csv(r'C:\Users\kazwe\OneDrive\Pulpit\IT\STUDIA\Semestr4\Zarzycki L\LABY4\temperature_and_energy_consumption.csv')

def zad2(road_traffic):
    """
    Wczytaj dane z pliku CSV lub innych źródeł danych.
    Zidentyfikuj i obsłuż brakujące wartości danych.
    Przeprowadź wstępną analizę danych w celu określenia odpowiedniej metody interpolacji.
    """
    print("Wczytaj dane z pliku CSV lub innych źródeł danych:")
    data_numeric = road_traffic.drop(columns=['Date', 'Time', 'Local Time (Sensor)', 'countlineName', 'direction'])
    print(data_numeric.describe())

    missing_values = data_numeric.isnull().sum()
    print("\nBrakujące:")
    print(missing_values)

    imputer = SimpleImputer(strategy='mean')
    road_traffic_imputed = pd.DataFrame(imputer.fit_transform(data_numeric), columns=data_numeric.columns)

    label_encoder = LabelEncoder()
    categorical_columns = road_traffic.select_dtypes(include=['object']).columns
    road_traffic_categorical = road_traffic[categorical_columns].copy()

    for col in categorical_columns:
        road_traffic_categorical[col] = label_encoder.fit_transform(road_traffic[col])
    road_traffic_processed = pd.concat([road_traffic_imputed, road_traffic_categorical], axis=1)

    correlation = road_traffic_imputed.corr()
    print("\nMacierz korelacji między typami pojazdów:")
    print(correlation)
    print("Duża zmienność sugeruje Interpolację wielomianową kawałkami")

    plt.figure(figsize=(12, 6))
    for i, column in enumerate(['Car', 'Pedestrian', 'Bus']):
        plt.subplot(1, 3, i + 1)
        plt.hist(road_traffic_imputed[column], bins=30)
        plt.title(f'Rozkład dla {column}')
    plt.tight_layout()

    return road_traffic_processed

def zad3():
    """
    Prognozowanie temperatury z danymi meteorologicznymi.
    Przygotuj funkcję do wygenerowania przykładowych danych do pliku weather_data.csv.
    Wczytaj dane z pliku CSV ( zgaduję że chodzi o weather_data.csv )
    Zidentyfikuj i obsłuż brakujące wartości danych.
    Wykorzystaj metodę interpolacji splajnów B-sklejanych do prognozowania temperatury na podstawie danych meteorologicznych z różnych stacji pomiarowych.
    """
    print("\nZadanie 3:")

    def generate_weather_data(num_stations, num_days):
        temperatures1 = np.array([-3, -1, 3, 0, 12, 14, 16, 31, 21, 10, 0, -4])
        np.random.seed(0)
        dates = pd.date_range(start='2023-01-01', periods=num_days)
        station_ids = ['Station_' + str(i) for i in range(1, num_stations + 1)]
        data = {station: [] for station in station_ids}
        for day in range(num_days):
            month = dates[day].month - 1
            temperature1 = temperatures1[month]
            for station in station_ids:
                temperature = temperature1 + np.random.uniform(low=-2, high=2) if station == 'Station_1' else temperature1 + np.random.uniform(low=-4, high=4)
                if day > 0 and np.random.rand() < 0.05:
                    temperature += np.random.uniform(low=-10, high=10)
                data[station].append(temperature)
        df = pd.DataFrame(data)
        df['Date'] = dates
        df = df[['Date'] + station_ids]
        df.to_csv('weather_data.csv', index=False)

    generate_weather_data(num_stations=5, num_days=15)

    data = pd.read_csv('weather_data.csv')

    print(data.head())
    print(data.info())

    data_numeric = data.drop(columns=['Date'])
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data_numeric), columns=data_numeric.columns)

    label_encoder = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != 'Date':
            data_imputed[col] = label_encoder.fit_transform(data[col])

    print("\nSprawdzone dane:")
    print(data_imputed.head())

    x = np.arange(len(data_imputed))
    x_new = np.linspace(0, len(data_imputed) - 1, 300)

    fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)  # 1 row, 5 columns

    for i, station in enumerate(data_imputed.columns):
        y = data_imputed[station]
        spl = splrep(x, y, s=0)
        y_new = splev(x_new, spl)

        axs[i].plot(x, y, 'o', label='Original')
        axs[i].plot(x_new, y_new, '-', label='B-spline Interpolation')
        axs[i].set_title(station)
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Temperature')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.suptitle('Temperature Forecast using B-spline Interpolation')
    plt.show()

def zad4(energy_consumption):
    """
    Prognozowanie zużycia energii.
    Wykorzystaj dane dotyczące zużycia energii z różnych źródeł do prognozowania przyszłego zużycia energii.
    Przeprowadź interpolację wielomianową lub inną metodę interpolacji
    dla danych zużycia energii w różnych sektorach i porównaj ich dokładność prognozowania
    """
    print("\nZadanie 4:")
    energy_consumption['time_n'] = pd.to_datetime(energy_consumption['time_n'])
    energy_consumption['day'] = (energy_consumption['time_n'] - energy_consumption['time_n'].min()).dt.days

    x = energy_consumption['day'].values
    y = energy_consumption['energy_consumption'].values

    energy_consumption_train = energy_consumption.copy()
    np.random.seed(42)
    energy_consumption_train.loc[np.random.choice(energy_consumption_train.index, size=int(
        len(energy_consumption_train) * 0.2)), 'energy_consumption'] = np.nan

    y_pred_poly = energy_consumption_train['energy_consumption'].interpolate(method='polynomial', order=3)
    y_pred_linear = energy_consumption_train['energy_consumption'].interpolate(method='linear')
    valid_idx = ~np.isnan(y) & ~np.isnan(y_pred_poly)
    mse_poly = mean_squared_error(y[valid_idx], y_pred_poly[valid_idx])
    mse_linear = mean_squared_error(y[valid_idx], y_pred_linear[valid_idx])

    print(f"MSE polynomial interpol: {mse_poly:.2f}")
    print(f"MSE linear interpol: {mse_linear:.2f}")
    print(f"Lepsze: {'Polynomial' if mse_poly < mse_linear else 'Linear'}")

    fig, axs = plt.subplots(3, figsize=(10, 15))

    axs[0].plot(energy_consumption['time_n'], y, 'o', label='Actual data')
    axs[0].set_title('Actual Energy Consumption Data')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Energy Consumption')
    axs[0].grid(True)

    axs[1].plot(energy_consumption['time_n'], y_pred_poly, '-', label='Polynomial Interpolation (3rd degree)')
    axs[1].set_title('Polynomial Interpolation (3rd degree)')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Energy Consumption')
    axs[1].grid(True)

    axs[2].plot(energy_consumption['time_n'], y_pred_linear, '--', label='Linear Interpolation')
    axs[2].set_title('Linear Interpolation')
    axs[2].set_xlabel('Date')
    axs[2].set_ylabel('Energy Consumption')
    axs[2].grid(True)

    plt.tight_layout()
    plt.suptitle('Comparison of Interpolation Methods for Energy Consumption Forecasting')
    plt.show()

def zad5(stocks_data):
    """
    Analiza trendów cen akcji.
    Zastosuj metodę interpolacji kubicznej Hermite'a do analizy trendów cen akcji z dużego zbioru danych giełdowych.
    Zidentyfikuj lokalne maksima i minima w cenach akcji oraz przewidywane zmiany trendów.
    Porównaj różne stopnie interpolacji i oceń ich wpływ na dokładność analizy trendów.
    """
    print("\nZadanie 5:")

    stocks_data['Date'] = pd.to_datetime(stocks_data['Date'])
    companies = [col for col in stocks_data.columns if col != 'Date']
    interpolation_degree = 200
    x = np.arange(len(stocks_data))

    fig, axs = plt.subplots(len(companies), 1, figsize=(12, 3 * len(companies)), sharex=True)

    for i, company in enumerate(companies):
        y = stocks_data[company].values
        dy_dx = np.gradient(y, x)
        spline = CubicHermiteSpline(x, y, dy_dx)
        x_fine = np.linspace(x.min(), x.max(), interpolation_degree)
        y_interp = spline(x_fine)
        max_idx = argrelextrema(y_interp, np.greater)[0]
        min_idx = argrelextrema(y_interp, np.less)[0]

        axs[i].plot(x, y, 'o', markersize=2, label='Original data')
        axs[i].plot(x_fine, y_interp, '-', label='Interpolation')
        axs[i].plot(x_fine[max_idx], y_interp[max_idx], 'go', markersize=4, label='Maxima')
        axs[i].plot(x_fine[min_idx], y_interp[min_idx], 'ro', markersize=4, label='Minima')
        axs[i].set_title(f'{company}')
        axs[i].grid(True)
        axs[i].legend(loc='upper right', fontsize='small')

    plt.xlabel('Time Index')
    fig.suptitle('Hermite Cubic Interpolation and Stock Price Trend Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def zad6(road_traffic, road_traffic_processed):
    """
    Prognozowanie ruchu ulicznego.
    Wykorzystaj dane z czujników ruchu ulicznego do prognozowania przyszłego ruchu na drogach.
    Zastosuj metodę interpolacji do analizy trendów w ruchu ulicznym i przewidywania przyszłych wzorców ruchu.
    Przeprowadź analizę porównawczą różnych metod interpolacji w kontekście prognozowania.
    """
    print("\nZadanie 6:")

    data = road_traffic.copy()

    transport_modes = ['Car', 'Pedestrian', 'Cyclist', 'Motorbike', 'Bus', 'OGV1', 'OGV2', 'LGV']

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
    axs = axs.flatten()

    for i, mode in enumerate(transport_modes):
        try:
            if 'Time' not in data.columns:
                print(f"Missing Time column. Available columns: {data.columns.tolist()}")
                raise ValueError("Missing Time column")

            road_traffic_agg = data.groupby("Time")[mode].mean().reset_index()

            if len(road_traffic_agg) <= 1:
                print(f"Not enough data points for interpolation for {mode}. Skipping.")
                ax = axs[i]
                ax.text(0.5, 0.5, f"Insufficient data for {mode}",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
                ax.set_title(f"{mode} - insufficient data")
                continue

            road_traffic_agg['time_idx'] = range(len(road_traffic_agg))
            x = road_traffic_agg['time_idx'].values
            y = road_traffic_agg[mode].values

            x_new = np.linspace(x.min(), x.max(), 500)

            ax = axs[i]
            ax.plot(x, y, 'o', label='Original data')

            if len(x) >= 2:
                linear_interp = interp1d(x, y, kind='linear', fill_value="extrapolate")
                y_linear = linear_interp(x_new)
                ax.plot(x_new, y_linear, label='Linear interpolation')

            if len(x) >= 4:
                cubic_interp = interp1d(x, y, kind='cubic', fill_value="extrapolate")
                y_cubic = cubic_interp(x_new)
                ax.plot(x_new, y_cubic, label='Cubic interpolation')

                spline_interp = CubicSpline(x, y)
                y_spline = spline_interp(x_new)
                ax.plot(x_new, y_spline, label='Spline interpolation')

            time_labels = road_traffic_agg['Time'].values
            ax.set_xticks(x)
            ax.set_xticklabels(time_labels, rotation=45)

            ax.set_title(f"{mode} - interpolation")
            ax.set_xlabel("Time")
            ax.set_ylabel("Average Count")
            ax.grid(True)
            ax.legend()

        except Exception as e:
            print(f"Error processing {mode}: {e}")
            ax = axs[i]
            ax.text(0.5, 0.5, f"Error processing {mode}: {str(e)}",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            ax.set_title(f"{mode} - error")

    plt.tight_layout()
    plt.suptitle("Comp of Interpolation Methods for Different Transport Modes", fontsize=16, y=1.02)
    plt.show()

if __name__ == "__main__":
    road_traffic_processed = zad2(road_traffic)
    zad3()
    zad4(energy_consumption)
    zad5(stocks_data)
    zad6(road_traffic, road_traffic_processed)