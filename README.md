# BigData


Lista 3.
1. Analiza i uruchomienie przykładów regresji.
2. Generowanie i przygotowanie danych mieszkaniowych do regresji.
3. Regresja liniowa (scikit-learn) do przewidywania cen mieszkań z oceną i wizualizacją.
4. Regresja wielomianowa dla zależności miesiąc–temperatura z oceną i wykresem.
5. Regresja grzbietowa i Lasso (scikit-learn) dla danych o temperaturze i zużyciu energii, porównanie metod.
6. Regresja SVR do przewidywania czasu przeżycia pacjentów, analiza parametrów i porównanie metod regresji.



Lista 4.
1. Analiza i uruchomienie przykładów interpolacji.
2. Przygotowanie danych z obsługą braków i wybór metody interpolacji.
3. Prognozowanie temperatury z interpolacją B-sklejaną na danych meteorologicznych.
4. Interpolacja danych zużycia energii do prognozowania w różnych sektorach.
5. Interpolacja kubiczna Hermite’a do analizy trendów cen akcji i identyfikacji punktów zwrotnych.
6. Prognozowanie ruchu ulicznego na podstawie danych z czujników przy użyciu interpolacji i porównanie metod.


Lista 5.
1. Wczytanie i wstępna analiza szeregu czasowego (braki, statystyki, wizualizacja).
2. Ekstrakcja cech z szeregu czasowego (statystyki, transformacje) i porównanie ich skuteczności.
3. Implementacja niestandardowych metod ekstrakcji cech (np. FFT, autokorelacja) w NumPy i analiza ich wydajności.



Lista 6.
1. PCA na zbiorze Breast Cancer – redukcja wymiarowości i wizualizacja danych medycznych.
2. t-SNE na zbiorze Digits – wizualizacja ręcznie pisanych cyfr w przestrzeni 2D.
3. NMF na zbiorze LFW – redukcja cech i wizualizacja komponentów twarzy jako obrazów.
4. SVD na zbiorze Wine – redukcja wymiarów i wizualizacja rozkładu odmian win.
5. LDA na zbiorze 20 Newsgroups – przetwarzanie tekstu, redukcja wymiarowości i identyfikacja tematów.



Lisat 7.
1. Klasyfikacja gatunków irysów za pomocą KNN i analiza metryk jakości modelu.
2. Regresja logistyczna na danych Breast Cancer do klasyfikacji nowotworów, ocena i interpretacja wyników.
3. Klasyfikacja cyfr w zbiorze MNIST za pomocą SVM i analiza błędów klasyfikacji.
4. Przewidywanie przeżycia pasażerów Titanica przy użyciu regresji logistycznej lub drzewa decyzyjnego.
5. Klasyfikacja obecności choroby serca z użyciem SVM i identyfikacja czynników ryzyka.


Lista 8.
1. K-means do segmentacji klientów banku według zachowań finansowych.
2. Mean Shift na danych Credit Card Default i porównanie z K-means.
3. Klastrowanie aglomeracyjne na danych Iris i interpretacja struktury klastrów.
4. GMM do grupowania danych Wine i analiza jakości klastrów.
5. DBSCAN na danych Olivetti Faces – grupowanie twarzy i identyfikacja anomalii.
6. Klastrowanie hierarchiczne danych Diabetes i analiza podgrup pacjentów.
7. Porównanie metod klastrowania (K-means, Mean Shift, GMM, DBSCAN) na danych o chorobach serca z użyciem metryk jakości.


Lisat 9.
1. Wczytanie zbioru danych Fashion MNIST z biblioteki Keras.
2. Normalizacja obrazów i zamiana etykiet na one-hot encoding.
3. Zbudowanie konwolucyjnej sieci neuronowej w Keras.
4. Trenowanie modelu z walidacją przez określoną liczbę epok.
5. Ocena skuteczności modelu na zbiorze testowym oraz wizualizacja wyników.
6. Predykcja nowych danych, analiza wyników za pomocą macierzy pomyłek i raportu klasyfikacji.
7. Formułowanie wniosków i propozycji ulepszeń na podstawie wydajności modelu.


Lista 10.
1. Klasyfikacja binarna raka piersi przy użyciu prostej sieci neuronowej w Keras na zbiorze Breast Cancer.
2. Klasyfikacja wieloklasowa irysów przy użyciu sieci neuronowej na zbiorze Iris.
3. Transfer learning z użyciem modelu VGG16 do klasyfikacji twarzy na bazie VGGFace.
4. Klasyfikacja obiektów w zdjęciach z użyciem głębokiej sieci na zbiorze COCO.
5. Segmentacja obrazów drogowych z wykorzystaniem sieci neuronowej na danych CamVid.



NOTES:

### Czym jest regresja?

**Regresja** to technika uczenia maszynowego i statystyki, której celem jest modelowanie związku między jedną zmienną zależną (np. ceną domu) a jedną lub więcej zmiennymi niezależnymi (np. powierzchnią, lokalizacją, liczbą pokoi). Model regresji przewiduje wartość liczbową (zmienną ciągłą).

---

Oczywiście! Oto Twoje uzupełnione notatki – rozszerzyłem je o część dotyczącą interpolacji i zadbałem o spójny styl formatowania:

---

### Rodzaje regresji:

#### 1. **Regresja wielomianowa**

To rozszerzenie regresji liniowej, które pozwala na dopasowanie **krzywej** zamiast linii prostej.

* Modeluje zależność jako funkcję wielomianową, np.
  \$y = \beta\_0 + \beta\_1 x + \beta\_2 x^2 + \ldots + \beta\_n x^n\$
* Przydatna, gdy dane mają **nieliniowy charakter**, ale nadal chcemy użyć regresji liniowej w przestrzeni cech przekształconych do wielomianu.

---

#### 2. **Regresja grzbietowa (Ridge Regression)**

To regresja liniowa z **regularyzacją L2**, czyli dodaniem kary za zbyt duże wartości wag:

* Funkcja kosztu:
  \$\text{MSE} + \alpha \sum\_{i} \beta\_i^2\$
* Chroni przed **przeuczeniem**, zwłaszcza przy współliniowości cech.
* Współczynniki są **zmniejszane**, ale nigdy dokładnie równe zeru.

---

#### 3. **Regresja LASSO (Least Absolute Shrinkage and Selection Operator)**

To regresja liniowa z **regularyzacją L1**:

* Funkcja kosztu:
  \$\text{MSE} + \alpha \sum\_{i} |\beta\_i|\$
* Powoduje, że niektóre współczynniki mogą być **dokładnie równe 0**, co oznacza **wybór cech (feature selection)**.
* Dobra, gdy mamy dużo cech, z których tylko część jest istotna.

---

#### 4. **SVR (Support Vector Regression)**

To wersja regresji oparta na metodzie wektorów nośnych:

* Nie minimalizuje klasycznego błędu, ale dopuszcza **margines błędu ε**, w którym błędy są ignorowane.
* Może używać **jąder (kerneli)**, aby modelować bardzo złożone, nieliniowe relacje.
* Dobrze sprawdza się przy danych z **szumem i odstającymi wartościami**.
* Funkcja kosztu bardziej "karze" punkty leżące daleko od marginesu.

---

### Interpolacja – podstawy i przykłady

Interpolacja to metoda szacowania wartości pomiędzy znanymi punktami danych. W przeciwieństwie do regresji, interpolacja **przechodzi dokładnie przez punkty danych**.

#### Klasyczne metody interpolacji:

* **Interpolacja liniowa**

  * Najprostsza forma – łączy sąsiednie punkty prostymi odcinkami.
  * Prosta, ale może tworzyć **niegładkie** funkcje.

* **Interpolacja wielomianowa**

  * Dopasowuje jeden wielomian do wszystkich punktów.
  * Może być **dokładna**, ale podatna na **niestabilność** (np. zjawisko Rungego) przy dużej liczbie punktów.

* **Interpolacja sklejanymi funkcjami (Splines)**

  * Używa **funkcji łamanych klasy \$C^1\$ lub \$C^2\$**, które są gładkie.
  * Popularne są **B-splines** i **spline kubiczne**.
  * Dobrze sprawdzają się przy dużych zbiorach punktów – zapewniają **gładkość i stabilność**.

* **Interpolacja Hermite’a (kubiczna)**

  * Uwzględnia nie tylko wartości, ale też **pochodne** w punktach.
  * Lepsze dopasowanie przy zachowaniu kształtu danych i spójności pierwszej pochodnej.

#### Analiza danych czasowych

FFT to algorytm do szybkiego obliczania transformaty Fouriera – narzędzia, które przekształca dane z dziedziny czasu do dziedziny częstotliwości.

Autokorelacja mierzy podobieństwo sygnału do samego siebie, ale z przesunięciem w czasie (lagiem). To forma korelacji, ale z tą samą serią danych przesuniętą względem siebie.

#### Definicje

Metody redukcji wymiarowości i analizy danych, często używane w eksploracji danych, wizualizacji czy modelowaniu.

---

### 1. **PCA (Principal Component Analysis)**

**Analiza głównych składowych**

* Metoda redukcji wymiarowości, która zamienia oryginalne cechy na nowe, nieskorelowane zmienne zwane **głównymi składowymi**.
* Główne składowe to kierunki maksymalnej wariancji danych.
* Pomaga uprościć dane, zachowując jak najwięcej informacji.
* Używana do wizualizacji, przyspieszania modeli i usuwania szumu.

---

### 2. **t-SNE (t-distributed Stochastic Neighbor Embedding)**

**Wizualizacja danych wysokowymiarowych**

* Metoda nieliniowej redukcji wymiarowości do 2D lub 3D.
* Zachowuje lokalną strukturę danych, czyli podobieństwa między punktami.
* Bardzo popularna do wizualizacji klastrów i grup w danych.
* Nie nadaje się do bardzo dużych zbiorów bez optymalizacji.

---

### 3. **NMF (Non-negative Matrix Factorization)**

**Nienegatywna faktoryzacja macierzy**

* Rozkłada macierz danych na dwie macierze z nieujemnymi wartościami.
* Użyteczna, gdy dane mają sens tylko jako wartości nieujemne (np. tekst, obrazy).
* Pomaga w interpretacji cech jako „tematów” lub „części”.
* Często stosowana w analizie tekstu (topic modeling) i analizie obrazów.

---

### 4. **SVD (Singular Value Decomposition)**

**Rozkład wartości osobliwych**

* Uniwersalna metoda rozkładu macierzy na trzy składniki (U, Σ, V^T).
* Podstawa dla PCA i innych metod redukcji wymiarowości.
* Używana w kompresji danych, rekomendacjach (np. filtracja kolaboratywna) i rozwiązywaniu układów równań.

---

### 5. **LDA (Latent Dirichlet Allocation)**

**Ukryty model Dirichleta (tematyczny)**

* Metoda modelowania tematów w dokumentach tekstowych.
* Zakłada, że dokumenty to mieszanki „tematów”, które są rozkładami słów.
* Pomaga automatycznie odkrywać tematy w dużych zbiorach tekstów.
* Popularna w przetwarzaniu języka naturalnego (NLP).


## uczenie maszynowe

### 1. **KNN (K-Nearest Neighbors)**

**K najbliższych sąsiadów**

* To **algorytm klasyfikacji (lub regresji)**, który klasyfikuje punkt na podstawie **klasy większości spośród jego K najbliższych sąsiadów** w przestrzeni cech.
* Nie wymaga fazy trenowania — klasyfikuje na bieżąco, porównując odległości między punktami.
* Wymaga zdefiniowania metryki odległości (np. euklidesowa).
* Prosty i intuicyjny, ale może być wolny przy dużych danych i wymaga odpowiedniego doboru K.

---

### 2. **SVM (Support Vector Machine)**

**Maszyna wektorów nośnych**

* To **algorytm klasyfikacji i regresji**, który szuka **hiperpłaszczyzny** maksymalizującej margines między klasami.
* W klasycznej wersji znajduje liniowy separator między klasami.
* Dzięki **jądrom (kernelom)** może modelować także złożone, nieliniowe granice decyzji.
* Skupia się na punktach krytycznych (wektorach nośnych), które wyznaczają granicę decyzji.
* Bardzo skuteczny przy wysokowymiarowych i złożonych danych.



## Klasteryzacja

klasteryzacja, czyli grupowania danych w zbiory (klastry) na podstawie ich podobieństwa.

---

### 1. **K-means**

* Algorytm, który dzieli dane na **K klastrów** (liczba K jest podana z góry).
* Działa iteracyjnie: przypisuje punkty do najbliższego centroidu (środka klastra), potem aktualizuje centroidy jako średnie punktów w klastrze.
* Prosty i szybki, ale wymaga podania liczby klastrów i dobrze działa przy kulistych, równych gęstościach klastrach.

---

### 2. **Mean Shift**

* Algorytm, który nie wymaga podania liczby klastrów z góry.
* Szuka **trybów gęstości** danych — przesuwa centroid w kierunku największej gęstości punktów (gradient wzwyż).
* Dobrze radzi sobie z klastrami o nieregularnych kształtach i różnej gęstości.
* Wolniejszy niż K-means.

---

### 3. **GMM (Gaussian Mixture Model)**

* Model probabilistyczny, który zakłada, że dane są mieszaniną kilku rozkładów normalnych (Gaussowskich).
* Każdy klaster jest modelem Gaussa o swojej średniej i kowariancji.
* Pozwala na miękkie przypisanie punktów do klastrów (prawdopodobieństwa przynależności).
* Bardziej elastyczny niż K-means, ale wymaga estymacji parametrów (EM – Expectation-Maximization).

---

### 4. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

* Algorytm oparty na gęstości punktów.
* Grupy tworzone są tam, gdzie jest dużo punktów blisko siebie (gęste obszary).
* Nie wymaga podawania liczby klastrów z góry.
* Potrafi wykrywać klastry o nieregularnych kształtach oraz odróżniać punkty szumu (outliers).
* Wrażliwy na parametry minimalnej liczby punktów i promienia.



