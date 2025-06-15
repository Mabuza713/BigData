import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
import pywt
from scipy.signal import find_peaks
import time
import os

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

output_dir = 'lab5_wykresy'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def log_to_file(file_path, message, mode='a'):
    with open(file_path, mode, encoding='utf-8') as f:
        f.write(message + '\n')

def zadanie_2():
    all_indices = {}
    
    file_paths = [
        'Bovespa_data_2025-03-31.csv',
        'CAC40_data_2025-03-31.csv',
        'DAX_data_2025-03-31.csv',
        'Dow_Jones_Industrial_Average_data_2025-03-31.csv',
        'FTSE100_data_2025-03-31.csv'
    ]
    
    for file_name in file_paths:
        path = os.path.join('dane', 'Indeksy_gieldowe_2015-2025-03-31', file_name)
        df = pd.read_csv(path)
        index_name = file_name.split('_')[0].replace('.csv', '')
        all_indices[index_name] = df
        
        log_file_path = os.path.join(output_dir, f'{index_name}_info.txt')
        log_to_file(log_file_path, f"=== Zadanie 2 ===\n")
        log_to_file(log_file_path, f"Dane z pliku: {file_name}")
    
    cleaned_indices = {}
    for name, df in all_indices.items():
        log_file_path = os.path.join(output_dir, f'{name}_info.txt')
        
        if 'Price' in df.columns and any(df['Price'] == 'Ticker'):
            df = df[~df['Price'].isin(['Ticker', 'Date'])]
            df = df.rename(columns={'Price': 'Date'})
            
            for col in ['Close', 'High', 'Low', 'Open']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values('Date')
        
        cleaned_indices[name] = df
        
        missing_values = df.isnull().sum().sum()
        log_to_file(log_file_path, f"Liczba rekordów: {len(df)}")
        log_to_file(log_file_path, f"Brakujące wartości: {missing_values}")
        
        if 'Close' in df.columns:
            close_col = 'Close'
        elif 'DJIA' in df.columns:
            close_col = 'DJIA'
        else:
            close_col = df.select_dtypes(include=[np.number]).columns[0]
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df[close_col])
        plt.title(f'Szereg czasowy {name}')
        plt.xlabel('Data')
        plt.ylabel('Wartość')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_zadanie2_szereg.png'))
        plt.close()
        
        if len(df) > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                log_to_file(log_file_path, f"\nStatystyka {col}:")
                log_to_file(log_file_path, f"Min: {df[col].min()}")
                log_to_file(log_file_path, f"Max: {df[col].max()}")
                log_to_file(log_file_path, f"Średnia: {df[col].mean()}")
                log_to_file(log_file_path, f"Odchylenie std: {df[col].std()}")
    
    return cleaned_indices

def zadanie_3(indices_data):
    results = {}
    for name, df in indices_data.items():
        if len(df) == 0:
            continue
            
        log_file_path = os.path.join(output_dir, f'{name}_info.txt')
        log_to_file(log_file_path, f"\n=== Zadanie 3 ===\n")
        
        if 'Close' in df.columns:
            value_col = 'Close'
        elif 'DJIA' in df.columns:
            value_col = 'DJIA'
        else:
            value_col = df.select_dtypes(include=[np.number]).columns[0]
            
        if 'Date' not in df.columns:
            continue
            
        df_copy = df.copy()
        df_copy.set_index('Date', inplace=True)
        
        recent_data = df_copy.iloc[-min(1000, len(df_copy)):].copy()
        
        stats_features = {
            'Średnia': recent_data[value_col].mean(),
            'Mediana': recent_data[value_col].median(),
            'Odchylenie standardowe': recent_data[value_col].std(),
            'Kwantyle': [
                recent_data[value_col].quantile(0.25),
                recent_data[value_col].quantile(0.75)
            ],
            'Skośność': recent_data[value_col].skew(),
            'Kurtoza': recent_data[value_col].kurt()
        }
        
        log_to_file(log_file_path, f"  Średnia: {stats_features['Średnia']:.2f}")
        log_to_file(log_file_path, f"  Mediana: {stats_features['Mediana']:.2f}")
        log_to_file(log_file_path, f"  Odchylenie std: {stats_features['Odchylenie standardowe']:.2f}")
        log_to_file(log_file_path, f"  Kwantyl 25%: {stats_features['Kwantyle'][0]:.2f}")
        log_to_file(log_file_path, f"  Kwantyl 75%: {stats_features['Kwantyle'][1]:.2f}")
        log_to_file(log_file_path, f"  Skośność: {stats_features['Skośność']:.2f}")
        log_to_file(log_file_path, f"  Kurtoza: {stats_features['Kurtoza']:.2f}")
        
        recent_data['MA_50'] = recent_data[value_col].rolling(window=50).mean()
        recent_data['MA_200'] = recent_data[value_col].rolling(window=200).mean()
        recent_data['Diff_1'] = recent_data[value_col].diff()
        recent_data['Volatility'] = recent_data[value_col].rolling(window=21).std()
        
        log_to_file(log_file_path, "\nCechy czasowe:")
        log_to_file(log_file_path, f"  Ostatnia MA_50: {recent_data['MA_50'].iloc[-1]:.2f}")
        log_to_file(log_file_path, f"  Ostatnia MA_200: {recent_data['MA_200'].iloc[-1]:.2f}")
        log_to_file(log_file_path, f"  Ostatnia wartość zmienności 21-dniowej: {recent_data['Volatility'].iloc[-1]:.2f}")
        
        plt.figure(figsize=(14, 10))
        
        plt.subplot(4, 1, 1)
        plt.plot(recent_data.index, recent_data[value_col], label=value_col)
        plt.plot(recent_data.index, recent_data['MA_50'], label='MA_50')
        plt.plot(recent_data.index, recent_data['MA_200'], label='MA_200')
        plt.title(f'{name} - Szereg z średnimi kroczącymi')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(4, 1, 2)
        plt.plot(recent_data.index, recent_data['Diff_1'])
        plt.title(f'{name} - Różnice pierwszego rzędu')
        plt.grid(True)
        
        plt.subplot(4, 1, 3)
        plt.plot(recent_data.index, recent_data['Volatility'])
        plt.title(f'{name} - Zmienność (21-dniowa)')
        plt.grid(True)
        
        try:
            filled_data = recent_data[value_col].ffill().dropna()
            if len(filled_data) > 252:
                decomposition = seasonal_decompose(filled_data, model='additive', period=252)
                
                plt.subplot(4, 1, 4)
                plt.plot(decomposition.trend)
                plt.title(f'{name} - Komponent trendu z dekompozycji')
                plt.grid(True)
                
                results[name] = {
                    'stats': stats_features,
                    'time_features': {
                        'last_MA50': recent_data['MA_50'].iloc[-1],
                        'last_MA200': recent_data['MA_200'].iloc[-1],
                        'volatility': recent_data['Volatility'].iloc[-1]
                    },
                    'decomposition': {
                        'trend': decomposition.trend.iloc[-1],
                        'seasonal': decomposition.seasonal.iloc[-1],
                        'residual': decomposition.resid.iloc[-1]
                    }
                }
                
                log_to_file(log_file_path, "\nDekompozycja sezonowa:")
                log_to_file(log_file_path, f"  Trend: {decomposition.trend.iloc[-1]:.2f}")
                log_to_file(log_file_path, f"  Sezonowość: {decomposition.seasonal.iloc[-1]:.2f}")
                log_to_file(log_file_path, f"  Reszta: {decomposition.resid.iloc[-1]:.2f}")
                
            else:
                plt.subplot(4, 1, 4)
                plt.text(0.5, 0.5, "Za mało danych do dekompozycji sezonowej", 
                        horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
                
                results[name] = {
                    'stats': stats_features,
                    'time_features': {
                        'last_MA50': recent_data['MA_50'].iloc[-1] if not pd.isna(recent_data['MA_50'].iloc[-1]) else None,
                        'last_MA200': recent_data['MA_200'].iloc[-1] if not pd.isna(recent_data['MA_200'].iloc[-1]) else None,
                        'volatility': recent_data['Volatility'].iloc[-1] if not pd.isna(recent_data['Volatility'].iloc[-1]) else None
                    },
                    'decomposition': None
                }
                
                log_to_file(log_file_path, "\nDekompozycja sezonowa: Za mało danych")
        except Exception as e:
            plt.subplot(4, 1, 4)
            plt.text(0.5, 0.5, f"Błąd dekompozycji: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            
            results[name] = {
                'stats': stats_features,
                'time_features': {
                    'last_MA50': recent_data['MA_50'].iloc[-1] if not pd.isna(recent_data['MA_50'].iloc[-1]) else None,
                    'last_MA200': recent_data['MA_200'].iloc[-1] if not pd.isna(recent_data['MA_200'].iloc[-1]) else None,
                    'volatility': recent_data['Volatility'].iloc[-1] if not pd.isna(recent_data['Volatility'].iloc[-1]) else None
                },
                'decomposition': None
            }
            
            log_to_file(log_file_path, f"\nBłąd dekompozycji: {str(e)}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_zadanie3_cechy.png'))
        plt.close()
    
    return results

def zadanie_4(indices_data):
    methods = ['Transformata Fouriera', 'Transformata falkowa', 'Autokorelacja', 'Wykrywanie ekstremów']
    results = {}
    
    for name, df in indices_data.items():
        if len(df) < 100:
            continue
            
        log_file_path = os.path.join(output_dir, f'{name}_info.txt')
        log_to_file(log_file_path, f"\n=== Zadanie 4 ===\n")
            
        if 'Close' in df.columns:
            value_col = 'Close'
        elif 'DJIA' in df.columns:
            value_col = 'DJIA'
        else:
            value_col = df.select_dtypes(include=[np.number]).columns[0]
            
        if 'Date' not in df.columns:
            continue
            
        df_copy = df.copy()
        df_copy.set_index('Date', inplace=True)
        
        ts = df_copy[value_col].dropna()
        if len(ts) < 100:
            continue
        
        log_to_file(log_file_path, "Porównanie czasów wykonania metod:")
        execution_times = []
        
        start_time = time.time()
        fft_result = np.fft.fft(ts.values)
        fourier_time = time.time() - start_time
        execution_times.append(fourier_time)
        log_to_file(log_file_path, f"  {methods[0]}: {fourier_time:.4f}s, {len(fft_result)} cech")
        
        start_time = time.time()
        try:
            sample_size = 2**int(np.log2(len(ts)))
            sample_data = ts.values[:sample_size]
            coeffs = pywt.wavedec(sample_data, 'haar', level=3)
            wavelet_features = []
            for coeff in coeffs:
                wavelet_features.extend(coeff)
            wavelet_time = time.time() - start_time
            execution_times.append(wavelet_time)
            log_to_file(log_file_path, f"  {methods[1]}: {wavelet_time:.4f}s, {len(wavelet_features)} cech")
        except Exception as e:
            wavelet_time = np.nan
            execution_times.append(wavelet_time)
            log_to_file(log_file_path, f"  {methods[1]}: błąd - {e}")
        
        start_time = time.time()
        n_lags = 40
        acf_result = acf(ts, nlags=n_lags)
        autocorr_time = time.time() - start_time
        execution_times.append(autocorr_time)
        log_to_file(log_file_path, f"  {methods[2]}: {autocorr_time:.4f}s, {len(acf_result)} cech")
        
        start_time = time.time()
        returns = ts.pct_change().dropna()
        peaks, _ = find_peaks(returns.values, height=0, distance=5)
        valleys, _ = find_peaks(-returns.values, height=0, distance=5)
        peak_features = [
            len(peaks),
            len(valleys),
            np.mean(returns.iloc[peaks]) if len(peaks) > 0 else 0,
            np.mean(returns.iloc[valleys]) if len(valleys) > 0 else 0,
            np.std(np.diff(peaks)) if len(peaks) > 1 else 0,
            np.std(np.diff(valleys)) if len(valleys) > 1 else 0
        ]
        peak_time = time.time() - start_time
        execution_times.append(peak_time)
        log_to_file(log_file_path, f"  {methods[3]}: {peak_time:.4f}s, {len(peak_features)} cech")
        
        results[name] = {
            'execution_times': execution_times,
            'methods': methods
        }
        
        plt.figure(figsize=(10, 6))
        plt.bar(methods, execution_times, color=['blue', 'green', 'orange', 'red'])
        plt.title(f'Porównanie czasu wykonania metod dla {name}')
        plt.xlabel('Metoda')
        plt.ylabel('Czas wykonania (sekundy)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_zadanie4_porownanie.png'))
        plt.close()
    
    return results

def zadanie_5(indices_features, performance_results):
    for name in indices_features.keys():
        log_file_path = os.path.join(output_dir, f'{name}_info.txt')
        log_to_file(log_file_path, f"\n=== Zadanie 5 ===\n")
        
        features = indices_features.get(name, {})
        if features:
            log_to_file(log_file_path, "Podsumowanie ekstrakcji cech:")
            if 'stats' in features:
                stats = features['stats']
                log_to_file(log_file_path, f"  Średnia: {stats['Średnia']:.2f}")
                log_to_file(log_file_path, f"  Odchylenie standardowe: {stats['Odchylenie standardowe']:.2f}")
                log_to_file(log_file_path, f"  Skośność: {stats['Skośność']:.2f}")
                log_to_file(log_file_path, f"  Kurtoza: {stats['Kurtoza']:.2f}")
            
            if 'time_features' in features and features['time_features']:
                time_feats = features['time_features']
                if time_feats['last_MA50'] is not None:
                    log_to_file(log_file_path, f"  Ostatnia wartość MA50: {time_feats['last_MA50']:.2f}")
                if time_feats['last_MA200'] is not None:
                    log_to_file(log_file_path, f"  Ostatnia wartość MA200: {time_feats['last_MA200']:.2f}")
                if time_feats['volatility'] is not None:
                    log_to_file(log_file_path, f"  Zmienność (21 dni): {time_feats['volatility']:.2f}")
            
        perf_results = performance_results.get(name, {})
        if perf_results and 'execution_times' in perf_results and 'methods' in perf_results:
            log_to_file(log_file_path, "\nWydajność metod ekstrakcji cech:")
            for method, time_value in zip(perf_results['methods'], perf_results['execution_times']):
                if not np.isnan(time_value):
                    log_to_file(log_file_path, f"  {method}: {time_value:.4f} sekund")
        
    plt.figure(figsize=(14, 10))
    all_methods = ['Transformata Fouriera', 'Transformata falkowa', 'Autokorelacja', 'Wykrywanie ekstremów']
    avg_times = [[] for _ in range(len(all_methods))]
    
    for result in performance_results.values():
        for i, time_value in enumerate(result.get('execution_times', [])):
            if not np.isnan(time_value):
                avg_times[i].append(time_value)
    
    avg_times = [np.mean(times) if times else np.nan for times in avg_times]
    
    plt.bar(all_methods, avg_times, color=['blue', 'green', 'orange', 'red'])
    plt.title('Średni czas wykonania metod ekstrakcji cech')
    plt.xlabel('Metoda')
    plt.ylabel('Czas wykonania (sekundy)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'zadanie5_srednie_czasy.png'))
    plt.close()

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    indices_data = zadanie_2()
    indices_features = zadanie_3(indices_data)
    performance_results = zadanie_4(indices_data)
    zadanie_5(indices_features, performance_results)

if __name__ == "__main__":
    main()