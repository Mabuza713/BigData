import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def zad2():
    tab = np.arange(1, 11)
    print(tab)
    maxx = np.max(tab)
    print(maxx)
    minn = np.min(tab)
    print(minn)
    sr = np.mean(tab)
    print(sr)
    odch = np.std(tab)
    print(odch)

def zad3():
    tab = np.arange(12).reshape(3, 4)
    print(tab)
    print(tab[1, 2])
    print(tab[:2, -2:])

def zad4():
    tab = np.arange(10)
    print(tab)
    tab = tab.reshape(2, 5)
    print(tab)
    tab_t = tab.T
    print(tab_t.shape)

def zad5():
    tab1 = np.arange(10).reshape(2, 5)
    tab2 = np.arange(10, 20).reshape(2, 5)
    print(tab1 + tab2)
    print(tab1*2)

def zad6():
    tab = np.arange(9).reshape(3, 3)
    w = np.array([1, 2, 3])
    tab += w
    print(tab)
    k = np.array([1, 2, 3]).reshape(3, 1)
    tab *= k
    print(tab)

def zad7():
    tab = np.random.rand(100)
    sum = np.sum(tab)
    print(sum)
    srednia = np.mean(tab)
    print(srednia)
    odchylenie = np.std(tab)
    print(odchylenie)
    skumulowana_suma = np.cumsum(tab)
    print(skumulowana_suma)
    skumulowany_iloczyn = np.cumprod(tab)
    print(skumulowany_iloczyn)

def zad8():
    tab = np.random.randint(1, 100, 10)
    tab_s = np.sort(tab)
    szuk = 50
    idx = np.searchsorted(tab_s, szuk)
    if idx < len(tab_s) and tab_s[idx] == szuk:
        print(szuk)
    else:
        print(idx)


def zad9():
    dane = pd.read_csv("lab2.csv")
    print(dane.shape)
    print(dane.head())

def zad10():
    dane = pd.read_csv("lab2.csv")
    print(dane[['Close', 'Volume']])
    war = dane['Close']==153.0
    print(war)
    kryt = dane.loc[war]
    print(kryt)

def zad11():
    dane = pd.read_csv('lab2.csv')
    dane = dane.dropna()
    print(dane)
    dane = dane.drop_duplicates()
    print(dane)
    dane['Close'] = pd.to_numeric(dane['Close'])
    print(dane)

def zad12():
    dane = pd.read_csv('lab2.csv')
    dane_g = dane.groupby('Close')
    print(dane_g.sum())
    print(dane_g['Low'].mean().round(2))
    print(dane_g.size())
    print(dane_g.agg({'Close' : 'mean','Low' : 'mean'}))

def zad13():
    dane = pd.read_csv("lab2.csv")
    dane['New'] = dane['Close'] + dane['Low']
    print(dane['New'])
    dane['New'] = dane['New'].apply(lambda x: x**2)
    print(dane['New'])
    dane['Symbol'] = dane['Symbol'].str.lower()
    print(dane['Symbol'])

def zad14():
    df = pd.read_csv('lab2.csv')
    plt.figure(figsize=(12, 6))
    for symbol, group in df.groupby('Symbol'):
        plt.plot(group['Date'], group['Close'], marker='o', linestyle='-', label=symbol)
    
    plt.title('Ceny zamknięcia akcji w czasie')
    plt.xlabel('Data')
    plt.ylabel('Cena zamknięcia ($)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('wykres_liniowy_cen.png')

    plt.figure(figsize=(12, 6))
    pivot_df = df.pivot(index='Date', columns='Symbol', values='Volume')
    pivot_df.plot(kind='bar', figsize=(12, 6))
    
    plt.title('Wolumen handlu akcjami')
    plt.xlabel('Data')
    plt.ylabel('Wolumen')
    plt.grid(axis='y')
    plt.legend(title='Symbol')
    plt.tight_layout()
    plt.savefig('wykres_slupkowy_wolumenu.png')

    plt.figure(figsize=(10, 8))
    symbols = df['Symbol'].unique()
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    for i, symbol in enumerate(symbols):
        symbol_data = df[df['Symbol'] == symbol]
        plt.scatter(symbol_data['Low'], symbol_data['High'], 
                   label=symbol, color=colors[i % len(colors)],
                   s=symbol_data['Volume']/10000, alpha=0.7)
    
    plt.title('Porównanie cen najwyższych i najniższych')
    plt.xlabel('Cena najniższa ($)')
    plt.ylabel('Cena najwyższa ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('wykres_rozrzutu.png')

    plt.figure(figsize=(12, 6))
    for symbol, group in df.groupby('Symbol'):
        plt.plot(group['Date'], group['High'] - group['Low'], marker='s', 
                linestyle='-', label=f'{symbol} (zakres cen)')
    
    plt.title('Zakres cen akcji (High-Low)')
    plt.xlabel('Data')
    plt.ylabel('Zakres cen ($)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('wykres_zakresow_cen.png')

def zad15():
    df = pd.read_csv('lab2.csv')
    stats_df = df.groupby('Symbol').agg({
        'Open': 'mean',
        'High': 'max',
        'Low': 'min',
        'Close': 'mean',
        'Volume': 'sum'
    }).reset_index()
    
    stats_df.columns = ['Symbol', 'Średnie_Otwarcie', 'Najwyższa_Cena', 
                       'Najniższa_Cena', 'Średnie_Zamknięcie', 'Suma_Wolumenu']
    
    merged_df = pd.merge(df, stats_df, on='Symbol', how='left')
    print(merged_df.head())
    pivot_close = df.pivot(index='Date', columns='Symbol', values='Close')
    pivot_volume = df.pivot(index='Date', columns='Symbol', values='Volume')
    
    print("Ceny zamknięcia według symbolu:")
    print(pivot_close)
    print("\nWolumen według symbolu:")
    print(pivot_volume)
    
    df_time = df.copy()
    df_time.set_index('Date', inplace=True)
    
    df_time.sort_index(inplace=True)
    daily_returns = df_time.groupby('Symbol')['Close'].pct_change() * 100
    
    df_time.index = pd.to_datetime(df_time.index)
    df_time['DayOfWeek'] = df_time.index.day_of_week  # 0=Monday, 6=Sunday
    
    day_map = {0: 'Poniedziałek', 1: 'Wtorek', 2: 'Środa', 
               3: 'Czwartek', 4: 'Piątek', 5: 'Sobota', 6: 'Niedziela'}
    df_time['DayOfWeek'] = df_time['DayOfWeek'].map(day_map)
    
    day_stats = df_time.groupby(['Symbol', 'DayOfWeek']).agg({
        'Close': ['mean', 'std'],
        'Volume': 'mean'
    })
    
    print("Dzienne zwroty:")
    print(daily_returns)
    print("\nStatystyki według dni tygodnia:")
    print(day_stats)

def zad16():
    x = np.arange(10)
    y = np.random.rand(10)
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Wykres liniowy')
    plt.show()
zad16()

def zad17():
    x = np.random.rand(10)
    y = np.random.rand(10)
    plt.scatter(x, y, color = 'r')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Wykres punktowy')
    plt.show()
zad17()

def zad18():
    kategorie = ['A', 'B', 'C', 'D']
    wartosci = [1, 2, 3, 4]
    plt.bar(kategorie, wartosci,color='y')
    plt.xlabel('Kategorie')
    plt.ylabel('Wartości')
    plt.title('Wykres słupkowy')
    plt.show()
zad18()

def zad19():
    dane = np.random.randn(100)
    plt.hist(dane, bins=60, alpha=0.5, color='y',edgecolor = 'black')
    plt.xlabel('Wartość')
    plt.ylabel('Częstotliwość')
    plt.title('Histogram danych')
    plt.show()
zad19()

def zad20():
    etykiety = ['A', 'B', 'C']
    wartosci = [1, 10, 100]
    colors = ['r', 'b', 'y', 'g']
    plt.pie(wartosci, labels=etykiety, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Wykres kołowy kategorii')
    plt.show()
zad20()

def zad21():
    x = np.linspace(0, 10, 100)
    y1 = np.cos(x)
    y2 = np.cos(2*x)
    y3 = np.cos(3*x)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(x, y1)
    axs[0, 0].set_title('Lin')
    axs[0, 1].scatter(x, y2)
    axs[0, 1].set_title('P')
    axs[1, 0].plot(x, y3)
    axs[1, 0].set_title('Lin 2')
    axs[1, 1].hist(y1, bins=20)
    axs[1, 1].set_title('H')
    plt.tight_layout()
    plt.show()

def zad22():
    df = pd.read_csv('lab2.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    print("Struktura danych:")
    print(df.info())
    print("\nPodgląd danych:")
    print(df.head())
    
    df['DailyChange'] = ((df['Close'] - df['Open']) / df['Open']) * 100
    avg_volume = df.groupby('Symbol')['Volume'].transform('mean')
    df['RelativeVolume'] = df['Volume'] / avg_volume
    df['TransactionValue'] = df['Close'] * df['Volume']
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#2c3e50')
    ax.set_facecolor('#2c3e50')
    
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    ax.title.set_color('white')
    

    symbols = df['Symbol'].unique()
    symbol_colors = {
        'AAPL': '#3498db',  
        'GOOG': '#e74c3c'   
    }
    
    def add_annotation(x, y, z, text, color='white'):
        ax.text(x, y, z, text, color=color, fontsize=9, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#34495e', alpha=0.8, edgecolor='#f1c40f'))
    
    for symbol in symbols:
        data = df[df['Symbol'] == symbol]
        volume_scaled = np.log1p(data['Volume']) * 20
        colors = ['y', 'g', 'b']
        
        scatter = ax.scatter(
            data['Open'],          # Oś X - Cena otwarcia
            data['Close'],         # Oś Y - Cena zamknięcia
            data['Volume'],        # Oś Z - Wolumen
            s=volume_scaled,
            c=colors,  
            alpha=0.8,  
            edgecolors='white',
            linewidth=0.5
        )
        
        dates_sorted = data.sort_values('Date')
        ax.plot(
            dates_sorted['Open'], 
            dates_sorted['Close'], 
            dates_sorted['Volume'], 
            color=symbol_colors[symbol], 
            alpha=0.5, 
            linestyle='-', 
            linewidth=1.5
        )
        
        first_day = data.iloc[0]
        last_day = data.iloc[-1]
        
        add_annotation(
            first_day['Open'], 
            first_day['Close'], 
            first_day['Volume'],
            f"{symbol} Start\n{first_day['Date'].strftime('%Y-%m-%d')}"
        )
        
        add_annotation(
            last_day['Open'], 
            last_day['Close'], 
            last_day['Volume'],
            f"{symbol} End\n{last_day['Date'].strftime('%Y-%m-%d')}"
        )
        
        max_vol_idx = data['Volume'].idxmax()
        max_vol = data.loc[max_vol_idx]
        
        add_annotation(
            max_vol['Open'], 
            max_vol['Close'], 
            max_vol['Volume'],
            f"{symbol} Max Vol\n{max_vol['Date'].strftime('%Y-%m-%d')}\n{max_vol['Volume']:,.0f}",
            color='#f1c40f'
        )
    x_range = np.linspace(df['Open'].min(), df['Open'].max(), 10)
    y_range = np.linspace(df['Close'].min(), df['Close'].max(), 10)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    
    z_mesh = np.ones_like(x_mesh) * df['Volume'].min()
    
    ax.plot_surface(
        x_mesh, x_mesh, z_mesh,  
        alpha=0.1,
        color='white',
        linewidth=0,
        antialiased=True
    )
    
    title_str = 'Zaawansowany Wykres 3D'
    
    ax.set_xlabel('Cena otwarcia ($)', fontsize=12, labelpad=15)
    ax.set_ylabel('Cena zamknięcia ($)', fontsize=12, labelpad=15)
    ax.set_zlabel('Wolumen obrotu', fontsize=12, labelpad=15)
    
    ax.set_title(f'{title_str}', fontsize=16, pad=30)
    ax.view_init(elev=30, azim=45)
    ax.grid(True, linestyle='--', alpha=0.3, color='white')
    
    legend_elements = []
    for symbol in symbols:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=symbol_colors[symbol], markersize=10,
                           label=symbol))
    
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='white', markersize=5,
                            label='Mały wolumen'))
    
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='white', markersize=15,
                            label='Duży wolumen'))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
              facecolor='#34495e', edgecolor='#f1c40f', framealpha=0.8, labelcolor='white')
    
    analysis_text = (
        f"Analiza danych:\n"
        f" • Liczba obserwacji: {len(df)}\n"
        f" • Zakres dat: {df['Date'].min().strftime('%Y-%m-%d')} - {df['Date'].max().strftime('%Y-%m-%d')}\n"
        f" • Maks. wolumen: {df['Volume'].max():,.0f} ({df.loc[df['Volume'].idxmax(), 'Symbol']})\n"
        f" • Maks. zmiana dzienna: {df['DailyChange'].max():.2f}% ({df.loc[df['DailyChange'].idxmax(), 'Symbol']})"
    )
    
    fig.text(0.02, 0.02, analysis_text, fontsize=9, color='white',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#34495e', alpha=0.8, edgecolor='#f1c40f'))
    fig.text(0.98, 0.02, "Vis", fontsize=10, color='#f1c40f',
            ha='right', va='bottom', style='italic')
    
    plt.tight_layout()
    plt.savefig('zaawansowany_wykres_3d.png', dpi=300, bbox_inches='tight', facecolor='#2c3e50')
zad22()