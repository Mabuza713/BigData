import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split


def task1_iris():
    """
    Zadanie 1: Analiza zbioru danych Iris
    - Wczytaj zbiór danych Iris z biblioteki scikit-learn
    - Przeprowadź podstawową eksplorację danych, taką jak wyświetlenie kilku pierwszych
      wierszy danych, informacje o kolumnach itp.
    - Przygotuj dane do budowy modeli klasyfikacji
    - Zbuduj klasyfikator k-najbliższych sąsiadów (KNN) do klasyfikacji gatunków irysów
    - Oceń jakość klasyfikatora za pomocą różnych metryk, takich jak dokładność, precyzja,
      czułość i specyficzność
    - Zinterpretuj wyniki i zidentyfikuj najlepszy model
    """
    print("===== Task 1: Iris Dataset Analysis =====")
    
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    iris_df = pd.DataFrame(X, columns=feature_names)
    iris_df['target'] = y
    
    print(iris_df.head())
    
    print("\nInformacje o zbiorze danych:")
    print(f"Liczba próbek: {X.shape[0]}")
    print(f"Liczba cech: {X.shape[1]}")
    print(f"Nazwy cech: {feature_names}")
    print(f"Klasy: {target_names}")
    print(f"Rozkład klas: {np.bincount(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Inicjalizacja modelu KNN z k=3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    # Obliczanie metryk klasyfikacji
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print("\nMetryki klasyfikacji dla KNN (k=3):")
    print(f"Dokładność: {accuracy:.4f}")
    print(f"Precyzja: {precision:.4f}")
    print(f"Czułość: {recall:.4f}")
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nMacierz pomyłek:")
    print(conf_matrix)
    
    return knn, X, y


def task2_breast_cancer():
    """
    Zadanie 2: Analiza zbioru danych Breast Cancer Wisconsin:
    - Wczytaj zbiór danych Breast Cancer Wisconsin
    - Przeprowadź czyszczenie i przygotowanie danych, usuwając brakujące wartości, skalując cechy itp.
    - Podziel zbiór danych na zestawy treningowy i testowy
    - Zbuduj model regresji logistycznej do klasyfikacji nowotworów jako łagodnych lub złośliwych
    - Oceń jakość klasyfikatora za pomocą różnych metryk oceny, takich jak dokładność, precyzja,
      czułość i specyficzność
    - Zinterpretuj wyniki i porównaj je z innymi modelami, jeśli to możliwe
    """
    print("\n===== Task 2: Breast Cancer Wisconsin Dataset Analysis =====")
    
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    feature_names = cancer.feature_names
    target_names = cancer.target_names
    cancer_df = pd.DataFrame(X, columns=feature_names)
    cancer_df['target'] = y
    
    print(cancer_df.head())
    
    print("\nInformacje o zbiorze danych:")
    print(f"Liczba próbek: {X.shape[0]}")
    print(f"Liczba cech: {X.shape[1]}")
    print(f"Nazwy cech: {feature_names[:5]}... (i więcej)")
    print(f"Klasy: {target_names}")
    print(f"Rozkład klas: {np.bincount(y)}")

    missing_values = cancer_df.isnull().sum().sum()
    print(f"\nLiczba brakujących wartości: {missing_values}")
    
    # Skalowanie cech
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    
    # Inicjalizacja modelu regresji logistycznej
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    
    # Trenowanie modelu
    logreg.fit(X_train, y_train)
    
    # Predykcja etykiet dla zbioru testowego
    y_pred = logreg.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    print("\nMetryki klasyfikacji dla regresji logistycznej:")
    print(f"Dokładność: {accuracy:.4f}")
    print(f"Precyzja: {precision:.4f}")
    print(f"Czułość: {recall:.4f}")
    print(f"Specyficzność: {specificity:.4f}")
    
    # ROC AUC
    y_pred_proba = logreg.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC: {roc_auc:.4f}")
    
    # Macierz pomyłek
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nMacierz pomyłek:")
    print(conf_matrix)
    
    # Porównanie z modelem SVM
    svm = SVC(probability=True)
    svm.fit(X_train, y_train)
    svm_y_pred = svm.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_y_pred)
    svm_y_pred_proba = svm.predict_proba(X_test)[:, 1]
    svm_roc_auc = roc_auc_score(y_test, svm_y_pred_proba)
    
    print("\nPorównanie z modelem SVM:")
    print(f"Dokładność LogReg: {accuracy:.4f} vs SVM: {svm_accuracy:.4f}")
    print(f"AUC LogReg: {roc_auc:.4f} vs SVM: {svm_roc_auc:.4f}")
    
    return logreg, X, y


def task3_digits():
    """
    Zadanie 3: Analiza zbioru danych Digits (MNIST):
    - Wczytaj zbiór danych Digits (MNIST) zawierający obrazy cyfr
    - Przeprowadź proces przygotowania danych, takich jak spłaszczenie obrazów i standaryzacja wartości pikseli
    - Zastosuj algorytm maszyny wektorów nośnych (SVM) do klasyfikacji cyfr
    - Oceń jakość klasyfikatora za pomocą metryk, takich jak dokładność i macierz pomyłek
    - Zinterpretuj wyniki, analizując, które cyfry są klasyfikowane najczęściej błędnie
    """
    print("\n===== Task 3: Digits (MNIST) Dataset Analysis =====")
    
    digits = load_digits()
    X = digits.data
    y = digits.target
    print(f"\nShape of the images: {digits.images.shape}")
    print(f"Shape of the flattened data: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Normalizacja danych - standaryzacja wartości pikseli
    X_scaled = X / 16.0
    
    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    
    # Inicjalizacja modelu SVM
    svm_model = SVC(kernel='rbf', gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    
    # Predykcja etykiet dla zbioru testowego
    y_pred = svm_model.predict(X_test)
    
    # Obliczanie metryk klasyfikacji
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nDokładność modelu SVM: {accuracy:.4f}")
    
    # Macierz pomyłek
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nMacierz pomyłek:")
    print(conf_matrix)
    
    error_rate_per_class = []
    for i in range(10):
        class_indices = np.where(y_test == i)[0]
        class_pred = y_pred[class_indices]
        error_rate = 1 - accuracy_score(y_test[class_indices], class_pred)
        error_rate_per_class.append(error_rate)
    
    # Sortowanie cyfr według częstości błędów
    sorted_indices = np.argsort(error_rate_per_class)[::-1]
    
    print("\nCyfry posortowane według częstości błędów (od najczęściej do najrzadziej błędnych):")
    for i, idx in enumerate(sorted_indices):
        print(f"Cyfra {idx}: {error_rate_per_class[idx]:.4f} błędów")
    
    return svm_model, X, y


def task4_titanic():
    """
    Zadanie 4: Analiza zbioru danych Titanic:
    - Wczytaj zbiór danych Titanic zawierający informacje o pasażerach statku
    - Przeprowadź analizę eksploracyjną danych, identyfikując istotne cechy dla przewidywania przeżycia
    - Przygotuj dane do budowy modeli klasyfikacji, usuwając brakujące wartości i kodując zmienne kategoryczne
    - Zbuduj modele klasyfikacji, takie jak regresja logistyczna lub drzewo decyzyjne, do przewidywania przeżycia pasażerów
    - Oceń jakość klasyfikatorów za pomocą metryk oceny, takich jak dokładność i krzywe ROC-AUC
    - Zinterpretuj wyniki i zidentyfikuj czynniki mające największy wpływ na przeżycie
    """
    print("\n===== Task 4: Titanic Dataset Analysis =====")
    
    try:
        titanic_df = pd.read_csv('Titanic-Dataset.csv')
        print(titanic_df.head())
        
        print("\nInformacje o zbiorze danych:")
        print(f"Liczba próbek: {titanic_df.shape[0]}")
        print(f"Liczba cech: {titanic_df.shape[1]}")
        print("\nTypy danych:")
        print(titanic_df.dtypes)
        
        missing_values = titanic_df.isnull().sum()
        print("\nBrakujące wartości:")
        print(missing_values[missing_values > 0])
        titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        
        # Uzupełnienie brakujących wartości
        titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
        titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)
        
        # Kodowanie zmiennych kategorycznych
        titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
        embarked_dummies = pd.get_dummies(titanic_df['Embarked'], prefix='Embarked')
        titanic_df = pd.concat([titanic_df, embarked_dummies], axis=1)
        titanic_df.drop('Embarked', axis=1, inplace=True)
        
        # Wybór cech i docelowej zmiennej
        X = titanic_df.drop('Survived', axis=1)
        y = titanic_df['Survived']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Inicjalizacja modelu regresji logistycznej
        logreg = LogisticRegression(max_iter=1000, random_state=42)
        logreg.fit(X_train, y_train)
        
        # Predykcja etykiet dla zbioru testowego
        y_pred = logreg.predict(X_test)
        
        # Obliczanie metryk klasyfikacji
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Specyficzność
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        
        # ROC AUC
        y_pred_proba = logreg.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print("\nMetryki klasyfikacji dla regresji logistycznej:")
        print(f"Dokładność: {accuracy:.4f}")
        print(f"Precyzja: {precision:.4f}")
        print(f"Czułość: {recall:.4f}")
        print(f"Specyficzność: {specificity:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        
        # Macierz pomyłek
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\nMacierz pomyłek:")
        print(conf_matrix)
        
        # Analiza współczynników modelu, aby zidentyfikować istotne cechy
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': logreg.coef_[0]
        })
        feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
        feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
        
        print("\nCzynniki mające największy wpływ na przeżycie:")
        print(feature_importance)
        
        return logreg, X, y
    
    except Exception as e:
        print(f"Wystąpił błąd podczas analizy danych Titanic: {e}")
        return None, None, None


def task5_heart_disease():
    """
    Zadanie 5: Analiza zbioru danych Heart Disease:
    - Wczytaj zbiór danych dotyczący chorób serca
    - Przeprowadź analizę eksploracyjną danych, identyfikując czynniki ryzyka chorób serca
    - Podziel zbiór danych na zestaw treningowy i testowy
    - Zbuduj klasyfikator SVM do przewidywania obecności choroby serca
    - Oceń jakość klasyfikatora za pomocą różnych metryk, takich jak dokładność, precyzja, czułość i specyficzność
    - Zinterpretuj wyniki, starając się zidentyfikować czynniki mające największy wpływ na ryzyko choroby serca
    """
    print("\n===== Task 5: Heart Disease Dataset Analysis =====")
    
    try:
        heart_df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
        print(heart_df.head())
        
        print("\nInformacje o zbiorze danych:")
        print(f"Liczba próbek: {heart_df.shape[0]}")
        print(f"Liczba cech: {heart_df.shape[1]}")
        print("\nTypy danych:")
        print(heart_df.dtypes)
        
        missing_values = heart_df.isnull().sum()
        missing_values_count = missing_values.sum()
        print(f"\nLiczba brakujących wartości: {missing_values_count}")
        
        # Statystyki opisowe
        print("\nStatystyki opisowe:")
        print(heart_df.describe())
        
        # Korelacja między cechami
        correlation = heart_df.corr()['DEATH_EVENT'].sort_values(ascending=False)
        print("\nKorelacja z DEATH_EVENT:")
        print(correlation)
        
        X = heart_df.drop('DEATH_EVENT', axis=1)
        y = heart_df['DEATH_EVENT']
        
        # Skalowanie cech
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)
        
        # Inicjalizacja modelu SVM
        svm_model = SVC(kernel='linear', probability=True, random_state=42)
        
        # Trenowanie modelu
        svm_model.fit(X_train, y_train)
        
        # Predykcja etykiet dla zbioru testowego
        y_pred = svm_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Specyficzność
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        
        # ROC AUC
        y_pred_proba = svm_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Wyświetlenie metryk
        print("\nMetryki klasyfikacji dla SVM:")
        print(f"Dokładność: {accuracy:.4f}")
        print(f"Precyzja: {precision:.4f}")
        print(f"Czułość: {recall:.4f}")
        print(f"Specyficzność: {specificity:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        
        # Macierz pomyłek
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\nMacierz pomyłek:")
        print(conf_matrix)
        
        return svm_model, X, y
    
    except Exception as e:
        print(f"Wystąpił błąd podczas analizy danych Heart Disease: {e}")
        return None, None, None


def main():
    task1_iris()
    task2_breast_cancer()
    task3_digits()
    task4_titanic()
    task5_heart_disease()


if __name__ == "__main__":
    main()