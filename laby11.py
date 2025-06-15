import os
import pandas as pd
from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

def zadanie1_world_bank():
    print("=== ZADANIE 1: Analiza danych World Bank ===")
    
    data_file_path = "API_SP.POP.TOTL_DS2_en_csv_v2_400708.csv"
    metadata_country_file_path = "Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_400708.csv"
    metadata_indicator_file_path = "Metadata_Indicator_API_SP.POP.TOTL_DS2_en_csv_v2_400708.csv"
    
    world_bank_data = pd.read_csv(data_file_path, skiprows=4)
    metadata_country = pd.read_csv(metadata_country_file_path)
    metadata_indicator = pd.read_csv(metadata_indicator_file_path)
    
    # Usuń niepotrzebne kolumny
    columns_to_drop = [col for col in world_bank_data.columns if 'Unnamed' in str(col)]
    if columns_to_drop:
        world_bank_data = world_bank_data.drop(columns_to_drop, axis=1)
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client["world_bank_database"]
    collection = db["countries_data"]
    metadata_country_collection = db["metadata_country"]
    metadata_indicator_collection = db["metadata_indicator"]
    
    collection.delete_many({})
    metadata_country_collection.delete_many({})
    metadata_indicator_collection.delete_many({})
    
    world_bank_data_dict = world_bank_data.to_dict("records")
    metadata_country_dict = metadata_country.to_dict("records")
    metadata_indicator_dict = metadata_indicator.to_dict("records")
    
    collection.insert_many(world_bank_data_dict)
    metadata_country_collection.insert_many(metadata_country_dict)
    metadata_indicator_collection.insert_many(metadata_indicator_dict)
    
    # Web scraping dla aktualnych danych populacji
    try:
        population_data_url = "https://www.worldometers.info/world-population/population-by-country/"
        response = requests.get(population_data_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        countries = []
        populations = []
        
        table = soup.find('table', id='example2')
        if table:
            rows = table.find_all('tr')[1:]
            for row in rows[:20]:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    country = cols[1].text.strip()
                    population_str = cols[2].text.strip().replace(',', '')
                    try:
                        population = int(population_str)
                        countries.append(country)
                        populations.append(population)
                    except ValueError:
                        continue
        
        population_df = pd.DataFrame({
            "Country": countries,
            "Population": populations
        })
        
        # Aktualizacja danych w MongoDB
        for index, row in population_df.iterrows():
            collection.update_one(
                {"Country Name": row['Country']},
                {"$set": {"Current_Population": row['Population']}},
                upsert=False
            )
    except Exception as e:
        print(f"Nie udało się pobrać danych z internetu: {e}")
    
    # Pobranie danych z MongoDB
    data = list(collection.find())
    df = pd.DataFrame(data)
    
    print(f"Wczytano {len(df)} rekordów z MongoDB")
    print("Kolumny:", list(df.columns))
    
    # Znajdź dostępne lata (kolumny numeryczne reprezentujące lata)
    year_columns = [col for col in df.columns if str(col).isdigit() and int(col) >= 1960]
    year_columns.sort(reverse=True)  # Sortuj od najnowszych
    
    print("Dostępne lata:", year_columns[:10])  # Pokaż tylko 10 najnowszych
    
    # Użyj danych z najnowszego dostępnego roku lub Current_Population
    current_pop_col = None
    if 'Current_Population' in df.columns:
        current_pop_col = 'Current_Population'
        print("Używam Current_Population z web scrapingu")
    else:
        # Znajdź najnowszy rok z danymi
        for year in year_columns:
            df[year] = pd.to_numeric(df[year], errors='coerce')
            if df[year].notna().sum() > 0:
                current_pop_col = year
                print(f"Używam danych z roku: {year}")
                break
    
    if current_pop_col is None:
        print("Brak dostępnych danych populacji")
        client.close()
        return
    
    # Filtruj dane z populacją
    df['Population_Data'] = pd.to_numeric(df[current_pop_col], errors='coerce')
    df_with_population = df[df['Population_Data'].notna() & (df['Population_Data'] > 0)]
    
    print(f"Krajów z danymi populacji: {len(df_with_population)}")
    
    # Sprawdzenie czy mamy dane do wizualizacji
    if len(df_with_population) == 0:
        print("Brak danych do wizualizacji")
        client.close()
        return
    
    # Weź top 15 krajów dla lepszej wizualizacji
    top_countries = df_with_population.nlargest(15, 'Population_Data')
    
    print(f"\nTop {len(top_countries)} krajów z największą populacją:")
    for idx, row in top_countries.iterrows():
        country_name = row.get('Country Name', 'Unknown')
        population = row.get('Population_Data', 0)
        print(f"{country_name}: {population:,.0f}")
    
    # Tworzenie wykresów
    plt.figure(figsize=(16, 12))
    
    # Wykres 1: Słupkowy poziomy
    plt.subplot(2, 2, 1)
    sns.barplot(data=top_countries.head(10), 
                x="Population_Data", 
                y="Country Name", 
                palette="viridis")
    plt.title("Top 10 krajów z największą populacją")
    plt.xlabel("Populacja")
    plt.ylabel("Kraj")
    plt.ticklabel_format(style='plain', axis='x')
    
    # Wykres 2: Pie chart dla top 5
    plt.subplot(2, 2, 2)
    top_5 = top_countries.head(5)
    plt.pie(top_5['Population_Data'], 
            labels=top_5['Country Name'], 
            autopct='%1.1f%%',
            startangle=90)
    plt.title("Udział top 5 krajów w populacji (spośród top 15)")
    
    # Wykres 3: Wykres słupkowy pionowy
    plt.subplot(2, 2, 3)
    plt.bar(range(len(top_countries.head(8))), 
            top_countries.head(8)['Population_Data'],
            color='skyblue')
    plt.title("Top 8 krajów - populacja")
    plt.xticks(range(len(top_countries.head(8))), 
               [name[:10] + '...' if len(name) > 10 else name 
                for name in top_countries.head(8)['Country Name']], 
               rotation=45)
    plt.ylabel("Populacja")
    plt.ticklabel_format(style='plain', axis='y')
    
    # Wykres 4: Histogram populacji
    plt.subplot(2, 2, 4)
    plt.hist(df_with_population['Population_Data'], bins=20, alpha=0.7, color='coral')
    plt.title("Rozkład populacji krajów")
    plt.xlabel("Populacja")
    plt.ylabel("Liczba krajów")
    plt.ticklabel_format(style='plain', axis='x')
    
    plt.tight_layout()
    
    # Utworzenie folderu na wykresy
    os.makedirs("wykresy", exist_ok=True)
    plt.savefig("wykresy/zadanie1_world_bank.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    client.close()
    print("Zadanie 1 zakończone pomyślnie!")

def zadanie3_imdb():
    print("=== ZADANIE 3: Analiza danych IMDB z agregacjami ===")
    
    imdb_data = pd.read_csv("imdb_top_1000.csv")
    print("Kolumny w zbiorze IMDB:", list(imdb_data.columns))
    
    # Czyszczenie i konwersja danych
    # Konwersja kolumn do odpowiednich typów
    imdb_data['IMDB_Rating'] = pd.to_numeric(imdb_data['IMDB_Rating'], errors='coerce')
    imdb_data['No_of_Votes'] = pd.to_numeric(imdb_data['No_of_Votes'], errors='coerce')
    imdb_data['Released_Year'] = pd.to_numeric(imdb_data['Released_Year'], errors='coerce')
    
    # Czyszczenie kolumny Gross - usunięcie przecinków i konwersja do liczb
    if 'Gross' in imdb_data.columns:
        imdb_data['Gross_Clean'] = imdb_data['Gross'].astype(str).str.replace(',', '').str.replace('$', '')
        imdb_data['Gross_Numeric'] = pd.to_numeric(imdb_data['Gross_Clean'], errors='coerce')
    else:
        imdb_data['Gross_Numeric'] = np.nan
    
    # Dodanie brakujących kolumn
    imdb_data['budget'] = imdb_data['Gross_Numeric'] * 0.6  # Szacowany budżet jako 60% przychodów
    imdb_data['budget'] = imdb_data['budget'].fillna(50000000)  # Domyślny budżet dla brakujących danych
    
    imdb_data['popularity_score'] = imdb_data['No_of_Votes'] / 10000
    imdb_data['popularity_score'] = imdb_data['popularity_score'].fillna(5.0)
    
    # Mapowanie kolumn
    column_mapping = {
        'Series_Title': 'title',
        'Released_Year': 'year', 
        'Genre': 'genre',
        'Director': 'director',
        'IMDB_Rating': 'rating',
        'Runtime': 'duration',
        'Gross_Numeric': 'revenue',
        'No_of_Votes': 'votes'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in imdb_data.columns:
            imdb_data = imdb_data.rename(columns={old_col: new_col})
    
    # Usunięcie wierszy z brakującymi danymi kluczowych
    imdb_data = imdb_data.dropna(subset=['rating', 'year'])
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client["imdb_database"]
    collection = db["movies"]
    collection.delete_many({})
    
    # Konwersja NaN na None dla MongoDB
    imdb_dict = imdb_data.where(pd.notnull(imdb_data), None).to_dict("records")
    collection.insert_many(imdb_dict)
    
    # Agregacja 1: Najlepsze filmy według roku
    pipeline_top_by_year = [
        {
            "$match": {"year": {"$ne": None}, "rating": {"$ne": None}}
        },
        {
            "$group": {
                "_id": "$year",
                "best_movie_rating": {"$max": "$rating"},
                "avg_rating": {"$avg": "$rating"},
                "movie_count": {"$sum": 1},
                "total_revenue": {"$sum": {"$ifNull": ["$revenue", 0]}}
            }
        },
        {"$sort": {"_id": -1}},
        {"$limit": 10}
    ]
    
    top_by_year = list(collection.aggregate(pipeline_top_by_year))
    print(f"Znaleziono {len(top_by_year)} lat z danymi")
    
    # Agregacja 2: Analiza gatunków
    pipeline_genre_analysis = [
        {
            "$match": {"genre": {"$ne": None}, "rating": {"$ne": None}}
        },
        {
            "$group": {
                "_id": "$genre",
                "avg_rating": {"$avg": "$rating"},
                "movie_count": {"$sum": 1},
                "total_revenue": {"$sum": {"$ifNull": ["$revenue", 0]}},
                "avg_budget": {"$avg": {"$ifNull": ["$budget", 0]}}
            }
        },
        {"$sort": {"avg_rating": -1}}
    ]
    
    genre_stats = list(collection.aggregate(pipeline_genre_analysis))
    print(f"Znaleziono {len(genre_stats)} gatunków")
    
    # Agregacja 3: Najlepsi reżyserowie
    pipeline_directors = [
        {
            "$match": {"director": {"$ne": None}, "rating": {"$ne": None}}
        },
        {
            "$group": {
                "_id": "$director",
                "avg_rating": {"$avg": "$rating"},
                "movie_count": {"$sum": 1},
                "total_revenue": {"$sum": {"$ifNull": ["$revenue", 0]}},
                "best_movie_rating": {"$max": "$rating"}
            }
        },
        {"$match": {"movie_count": {"$gte": 2}}},
        {"$sort": {"avg_rating": -1}},
        {"$limit": 10}
    ]
    
    top_directors = list(collection.aggregate(pipeline_directors))
    print(f"Znaleziono {len(top_directors)} reżyserów z wieloma filmami")
    
    # Agregacja 4: Rentowność (poprawiona)
    pipeline_profitability = [
        {
            "$match": {
                "revenue": {"$ne": None, "$gt": 0},
                "budget": {"$ne": None, "$gt": 0}
            }
        },
        {
            "$addFields": {
                "profit": {"$subtract": ["$revenue", "$budget"]},
                "profit_margin": {
                    "$multiply": [
                        {"$divide": [{"$subtract": ["$revenue", "$budget"]}, "$budget"]},
                        100
                    ]
                }
            }
        },
        {
            "$match": {"profit": {"$gt": 0}}
        },
        {
            "$sort": {"profit_margin": -1}
        },
        {
            "$limit": 10
        }
    ]
    
    try:
        most_profitable = list(collection.aggregate(pipeline_profitability))
        print(f"Znaleziono {len(most_profitable)} rentownych filmów")
    except Exception as e:
        print(f"Błąd w agregacji rentowności: {e}")
        most_profitable = []
    
    # Pobieranie danych do wizualizacji
    all_movies = list(collection.find())
    df_viz = pd.DataFrame(all_movies)
    
    # Konwersja kolumn do odpowiednich typów
    numeric_columns = ['rating', 'year', 'revenue', 'budget', 'votes', 'popularity_score']
    for col in numeric_columns:
        if col in df_viz.columns:
            df_viz[col] = pd.to_numeric(df_viz[col], errors='coerce')
    
    # Czyszczenie danych do wizualizacji
    df_viz = df_viz.dropna(subset=['rating', 'year'])
    
    plt.figure(figsize=(20, 15))
    
    # Wykres 1: Oceny według roku
    plt.subplot(3, 3, 1)
    yearly_ratings = df_viz.groupby('year')['rating'].mean()
    plt.plot(yearly_ratings.index, yearly_ratings.values, marker='o')
    plt.title('Średnia ocena filmów według roku')
    plt.xlabel('Rok')
    plt.ylabel('Średnia ocena')
    plt.grid(True, alpha=0.3)
    
    # Wykres 2: Liczba filmów według roku
    plt.subplot(3, 3, 2)
    year_counts = df_viz['year'].value_counts().sort_index()
    plt.bar(year_counts.index, year_counts.values, alpha=0.7)
    plt.title('Liczba filmów według roku')
    plt.xlabel('Rok')
    plt.ylabel('Liczba filmów')
    
    # Wykres 3: Top gatunki
    plt.subplot(3, 3, 3)
    if 'genre' in df_viz.columns:
        genre_counts = df_viz['genre'].value_counts().head(8)
        plt.barh(range(len(genre_counts)), genre_counts.values)
        plt.yticks(range(len(genre_counts)), genre_counts.index)
        plt.title('Najpopularniejsze gatunki')
        plt.xlabel('Liczba filmów')
    
    # Wykres 4: Rozkład ocen
    plt.subplot(3, 3, 4)
    plt.hist(df_viz['rating'], bins=20, alpha=0.7, color='skyblue')
    plt.title('Rozkład ocen IMDB')
    plt.xlabel('Ocena')
    plt.ylabel('Liczba filmów')
    
    # Wykres 5: Ocena vs liczba głosów
    plt.subplot(3, 3, 5)
    if 'votes' in df_viz.columns:
        plt.scatter(df_viz['votes'], df_viz['rating'], alpha=0.6)
        plt.title('Ocena vs Liczba głosów')
        plt.xlabel('Liczba głosów')
        plt.ylabel('Ocena')
        plt.xscale('log')
    
    # Wykres 6: Top reżyserowie
    plt.subplot(3, 3, 6)
    if 'director' in df_viz.columns:
        director_ratings = df_viz.groupby('director')['rating'].mean().nlargest(8)
        plt.barh(range(len(director_ratings)), director_ratings.values)
        plt.yticks(range(len(director_ratings)), 
                   [name[:20] + '...' if len(name) > 20 else name for name in director_ratings.index])
        plt.title('Najwyżej oceniani reżyserowie')
        plt.xlabel('Średnia ocena')
    
    # Wykres 7: Budżet vs przychód
    plt.subplot(3, 3, 7)
    if 'revenue' in df_viz.columns and 'budget' in df_viz.columns:
        clean_data = df_viz.dropna(subset=['revenue', 'budget'])
        if len(clean_data) > 0:
            plt.scatter(clean_data['budget'], clean_data['revenue'], alpha=0.6)
            plt.title('Budżet vs Przychód')
            plt.xlabel('Budżet')
            plt.ylabel('Przychód')
            plt.ticklabel_format(style='scientific', scilimits=(0,0))
    
    # Wykres 8: Filmy według dekad
    plt.subplot(3, 3, 8)
    df_viz['decade'] = (df_viz['year'] // 10) * 10
    decade_counts = df_viz['decade'].value_counts().sort_index()
    plt.bar(decade_counts.index, decade_counts.values, alpha=0.7, color='coral')
    plt.title('Liczba filmów według dekad')
    plt.xlabel('Dekada')
    plt.ylabel('Liczba filmów')
    
    # Wykres 9: Korelacja
    plt.subplot(3, 3, 9)
    correlation_cols = ['rating', 'year', 'votes']
    if 'revenue' in df_viz.columns:
        correlation_cols.append('revenue')
    if 'budget' in df_viz.columns:
        correlation_cols.append('budget')
    
    corr_data = df_viz[correlation_cols].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, cbar=True)
    plt.title('Korelacja między zmiennymi')
    
    plt.tight_layout()
    
    os.makedirs("wykresy", exist_ok=True)
    plt.savefig("wykresy/zadanie3_imdb.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Wyświetlenie wyników agregacji
    print("\n=== WYNIKI AGREGACJI ===")
    print(f"Lata z najlepszymi filmami: {len(top_by_year)}")
    print(f"Analizowane gatunki: {len(genre_stats)}")
    print(f"Top reżyserowie: {len(top_directors)}")
    print(f"Rentowne filmy: {len(most_profitable)}")
    
    client.close()
    print("Zadanie 3 zakończone pomyślnie!")

# Reszta funkcji pozostaje bez zmian...
def zadanie2_titanic():
    print("=== ZADANIE 2: Analiza danych Titanic z operacjami CRUD ===")
    
    titanic_data = pd.read_csv("titanic.csv")
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client["titanic_database"]
    collection = db["passengers"]
    collection.delete_many({})
    
    titanic_dict = titanic_data.fillna("Unknown").to_dict("records")
    
    sample_passenger = {
        "PassengerId": 99999,
        "Survived": 1,
        "Pclass": 1,
        "Name": "Test, John",
        "Sex": "male",
        "Age": 35,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100.0,
        "Embarked": "S",
        "AddedDate": datetime.now()
    }
    
    collection.insert_one(sample_passenger)
    collection.insert_many(titanic_dict)
    
    collection.update_one(
        {"PassengerId": 99999},
        {"$set": {"Fare": 150.0, "UpdatedDate": datetime.now()}}
    )
    
    collection.update_many(
        {"Age": {"$lt": 18}},
        {"$set": {"AgeCategory": "Child"}}
    )
    
    collection.update_many(
        {"Age": {"$gte": 18, "$lt": 65}},
        {"$set": {"AgeCategory": "Adult"}}
    )
    
    collection.update_many(
        {"Age": {"$gte": 65}},
        {"$set": {"AgeCategory": "Senior"}}
    )
    
    collection.delete_one({"PassengerId": 99999})
    
    analysis_data = list(collection.find())
    df_analysis = pd.DataFrame(analysis_data)
    
    numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
    for col in numeric_cols:
        if col in df_analysis.columns:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 3, 1)
    survival_by_sex = df_analysis.groupby(['Sex', 'Survived']).size().unstack()
    survival_by_sex.plot(kind='bar', ax=plt.gca())
    plt.title('Przeżycie według płci')
    plt.xlabel('Płeć')
    plt.ylabel('Liczba pasażerów')
    plt.legend(['Nie przeżył', 'Przeżył'])
    plt.xticks(rotation=0)
    
    plt.subplot(2, 3, 2)
    df_analysis['Age'].hist(bins=30, alpha=0.7)
    plt.title('Rozkład wieku pasażerów')
    plt.xlabel('Wiek')
    plt.ylabel('Liczba pasażerów')
    
    plt.subplot(2, 3, 3)
    survival_by_class = df_analysis.groupby(['Pclass', 'Survived']).size().unstack()
    survival_by_class.plot(kind='bar', ax=plt.gca())
    plt.title('Przeżycie według klasy')
    plt.xlabel('Klasa')
    plt.ylabel('Liczba pasażerów')
    plt.legend(['Nie przeżył', 'Przeżył'])
    plt.xticks(rotation=0)
    
    plt.subplot(2, 3, 4)
    df_analysis['Fare'].hist(bins=50, alpha=0.7)
    plt.title('Rozkład cen biletów')
    plt.xlabel('Cena biletu')
    plt.ylabel('Liczba pasażerów')
    plt.xlim(0, 200)
    
    plt.subplot(2, 3, 5)
    correlation_data = df_analysis[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr()
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0)
    plt.title('Korelacja między zmiennymi')
    
    plt.subplot(2, 3, 6)
    survival_by_embarked = df_analysis.groupby(['Embarked', 'Survived']).size().unstack()
    survival_by_embarked.plot(kind='bar', ax=plt.gca())
    plt.title('Przeżycie według portu zaokrętowania')
    plt.xlabel('Port (C=Cherbourg, Q=Queenstown, S=Southampton)')
    plt.ylabel('Liczba pasażerów')
    plt.legend(['Nie przeżył', 'Przeżył'])
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    os.makedirs("wykresy", exist_ok=True)
    plt.savefig("wykresy/zadanie2_titanic.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    client.close()

def generate_sample_weather_data(n_records=1000):
    np.random.seed(42)
    cities = ['Warsaw', 'Krakow', 'Gdansk', 'Wroclaw', 'Poznan', 'Lodz', 'Szczecin', 'Lublin']
    
    start_date = datetime.now() - timedelta(days=5*365)
    dates = [start_date + timedelta(days=x) for x in range(n_records)]
    
    weather_data = []
    for i, date in enumerate(dates):
        day_of_year = date.timetuple().tm_yday
        base_temp = 10 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        record = {
            'date': date,
            'city': random.choice(cities),
            'temperature': round(base_temp + random.normalvariate(0, 5), 1),
            'humidity': random.randint(30, 90),
            'precipitation': round(max(0, random.normalvariate(2, 3)), 1),
            'wind_speed': round(random.uniform(0, 25), 1),
            'pressure': round(random.normalvariate(1013, 15), 1),
            'year': date.year,
            'month': date.month,
            'season': get_season(date.month)
        }
        weather_data.append(record)
    
    return pd.DataFrame(weather_data)

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

def fetch_climate_change_data():
    climate_data = {
        'global_temp_increase': 1.2,
        'co2_levels': 415,
        'sea_level_rise': 3.3,
        'arctic_ice_decline': 13,
        'extreme_weather_events': '+23% since 1980'
    }
    return climate_data

def zadanie4_dane_meteorologiczne():
    print("=== ZADANIE 4: Analiza danych meteorologicznych ===")
    
    weather_data = generate_sample_weather_data(1000)
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client["weather_database"]
    collection = db["weather_data"]
    collection.delete_many({})
    
    weather_dict = weather_data.to_dict("records")
    collection.insert_many(weather_dict)
    
    climate_data = fetch_climate_change_data()
    
    if climate_data:
        climate_collection = db["climate_change"]
        climate_collection.delete_many({})
        climate_record = {
            "timestamp": datetime.now(),
            "data": climate_data,
            "source": "Climate Change Knowledge Portal (simulated)"
        }
        climate_collection.insert_one(climate_record)
    
    pipeline_temp_trends = [
        {
            "$group": {
                "_id": {
                    "city": "$city",
                    "year": "$year"
                },
                "avg_temperature": {"$avg": "$temperature"},
                "max_temperature": {"$max": "$temperature"},
                "min_temperature": {"$min": "$temperature"}
            }
        },
        {
            "$sort": {"_id.year": 1}
        }
    ]
    
    temp_trends = list(collection.aggregate(pipeline_temp_trends))
    
    pipeline_precipitation = [
        {
            "$group": {
                "_id": {
                    "season": "$season",
                    "year": "$year"
                },
                "total_precipitation": {"$sum": "$precipitation"},
                "avg_precipitation": {"$avg": "$precipitation"},
                "rainy_days": {"$sum": {"$cond": [{"$gt": ["$precipitation", 0]}, 1, 0]}}
            }
        },
        {
            "$sort": {"_id.year": 1}
        }
    ]
    
    precipitation_trends = list(collection.aggregate(pipeline_precipitation))
    
    extreme_weather = list(collection.find({
        "$or": [
            {"temperature": {"$gt": 35}},
            {"temperature": {"$lt": -20}},
            {"precipitation": {"$gt": 50}},
            {"wind_speed": {"$gt": 20}}
        ]
    }))
    
    yearly_stats = list(collection.aggregate([
        {
            "$group": {
                "_id": "$year",
                "avg_temp": {"$avg": "$temperature"},
                "total_precip": {"$sum": "$precipitation"},
                "extreme_events": {
                    "$sum": {
                        "$cond": [
                            {
                                "$or": [
                                    {"$gt": ["$temperature", 35]},
                                    {"$lt": ["$temperature", -20]},
                                    {"$gt": ["$precipitation", 50]}
                                ]
                            },
                            1,
                            0
                        ]
                    }
                }
            }
        },
        {"$sort": {"_id": 1}}
    ]))
    
    all_weather = list(collection.find())
    df_weather = pd.DataFrame(all_weather)
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    
    plt.figure(figsize=(20, 15))
    
    plt.subplot(3, 3, 1)
    monthly_temp = df_weather.groupby([df_weather['date'].dt.year, df_weather['date'].dt.month])['temperature'].mean()
    monthly_temp.plot()
    plt.title('Trend temperatury w czasie')
    plt.xlabel('Rok-Miesiąc')
    plt.ylabel('Temperatura (°C)')
    
    plt.subplot(3, 3, 2)
    sns.boxplot(data=df_weather, x='season', y='temperature')
    plt.title('Rozkład temperatur według pór roku')
    plt.xticks(rotation=45)
    
    plt.subplot(3, 3, 3)
    city_precip = df_weather.groupby('city')['precipitation'].sum()
    city_precip.plot(kind='bar')
    plt.title('Łączne opady według miast')
    plt.xticks(rotation=45)
    plt.ylabel('Opady (mm)')
    
    plt.subplot(3, 3, 4)
    weather_corr = df_weather[['temperature', 'humidity', 'precipitation', 'wind_speed', 'pressure']].corr()
    sns.heatmap(weather_corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Korelacja zmiennych pogodowych')
    
    plt.subplot(3, 3, 5)
    df_weather['humidity'].hist(bins=30, alpha=0.7)
    plt.title('Rozkład wilgotności')
    plt.xlabel('Wilgotność (%)')
    plt.ylabel('Liczba pomiarów')
    
    plt.subplot(3, 3, 6)
    city_temps = df_weather.groupby('city')['temperature'].mean()
    sns.barplot(x=city_temps.index, y=city_temps.values)
    plt.title('Średnie temperatury według miast')
    plt.xticks(rotation=45)
    plt.ylabel('Temperatura (°C)')
    
    plt.subplot(3, 3, 7)
    monthly_precip = df_weather.groupby([df_weather['date'].dt.year, df_weather['date'].dt.month])['precipitation'].sum()
    monthly_precip.plot()
    plt.title('Trend opadów w czasie')
    plt.xlabel('Rok-Miesiąc')
    plt.ylabel('Opady (mm)')
    
    plt.subplot(3, 3, 8)
    df_weather['wind_speed'].hist(bins=30, alpha=0.7)
    plt.title('Rozkład prędkości wiatru')
    plt.xlabel('Prędkość wiatru (m/s)')
    plt.ylabel('Liczba pomiarów')
    
    plt.subplot(3, 3, 9)
    plt.scatter(df_weather['temperature'], df_weather['precipitation'], alpha=0.6)
    plt.title('Temperatura vs Opady')
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('Opady (mm)')
    
    plt.tight_layout()
    
    os.makedirs("wykresy", exist_ok=True)
    plt.savefig("wykresy/zadanie4_weather.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    client.close()

def main():
    print("LABORATORIUM: INTEGRACJA MONGODB Z ANALIZĄ DANYCH W PYTHONIE")
    
    while True:
        print("\nWybierz zadanie do wykonania:")
        print("1. Zadanie 1 - Analiza danych World Bank")
        print("2. Zadanie 2 - Operacje CRUD na danych Titanic")
        print("3. Zadanie 3 - Analiza danych IMDB z agregacjami")
        print("4. Zadanie 4 - Dane meteorologiczne i zmiany klimatyczne")
        print("5. Uruchom wszystkie zadania")
        print("0. Wyjście")
        
        choice = input("\nPodaj numer zadania (0-5): ").strip()
        
        try:
            if choice == '1':
                zadanie1_world_bank()
            elif choice == '2':
                zadanie2_titanic()
            elif choice == '3':
                zadanie3_imdb()
            elif choice == '4':
                zadanie4_dane_meteorologiczne()
            elif choice == '5':
                zadanie1_world_bank()
                zadanie2_titanic()
                zadanie3_imdb()
                zadanie4_dane_meteorologiczne()
            elif choice == '0':
                break
            else:
                print("Nieprawidłowy wybór. Wybierz liczbę od 0 do 5.")
        except Exception as e:
            print(f"Wystąpił błąd podczas wykonywania zadania: {e}")
            print("Sprawdź czy MongoDB jest uruchomiony i czy wszystkie pliki CSV są dostępne.")
        
        input("\nNaciśnij Enter, aby kontynuować...")

if __name__ == "__main__":
    main()