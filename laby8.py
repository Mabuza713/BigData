import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

print("=== 1. Wczytanie i przygotowanie danych ===")
data = pd.read_csv('Card.csv')
print(f"Rozmiar danych: {data.shape}")
print(f"Kolumny: {data.columns.tolist()}")

print(f"\nBrakujące wartości:")
print(data.isnull().sum())

print(f"\nPodstawowe statystyki:")
print(data.describe())

print("\n=== 2. Algorytm K-means ===")

financial_features = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 
                     'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 
                     'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[financial_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(data_scaled)

data['kmeans_cluster'] = kmeans_labels

kmeans_silhouette = silhouette_score(data_scaled, kmeans_labels)
kmeans_calinski = calinski_harabasz_score(data_scaled, kmeans_labels)
kmeans_davies = davies_bouldin_score(data_scaled, kmeans_labels)

print(f"K-means - Liczba klastrów: 3")
print(f"Silhouette Score: {kmeans_silhouette:.4f}")
print(f"Calinski-Harabasz Score: {kmeans_calinski:.4f}")
print(f"Davies-Bouldin Score: {kmeans_davies:.4f}")

print("\n=== 3. Algorytm Mean Shift ===")

ms = MeanShift(bandwidth=1.0, cluster_all=True)
ms_labels = ms.fit_predict(data_scaled)
n_clusters_ms = len(set(ms_labels)) - (1 if -1 in ms_labels else 0)

data['meanshift_cluster'] = ms_labels

if n_clusters_ms > 1:
    ms_silhouette = silhouette_score(data_scaled, ms_labels)
    ms_calinski = calinski_harabasz_score(data_scaled, ms_labels)
    ms_davies = davies_bouldin_score(data_scaled, ms_labels)
    
    print(f"Mean Shift - Liczba klastrów: {n_clusters_ms}")
    print(f"Silhouette Score: {ms_silhouette:.4f}")
    print(f"Calinski-Harabasz Score: {ms_calinski:.4f}")
    print(f"Davies-Bouldin Score: {ms_davies:.4f}")
else:
    print(f"Mean Shift - Liczba klastrów: {n_clusters_ms} (za mało klastrów do oceny)")

print("\n=== 4. Porównanie wyników z K-means ===")

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
plt.title('Grupowanie za pomocą metody K-means')
plt.xlabel('Pierwsza składowa główna')
plt.ylabel('Druga składowa główna')

plt.subplot(1, 3, 2)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=ms_labels, cmap='viridis', alpha=0.6)
plt.title('Metoda przesunięcia średniej (Mean Shift)')
plt.xlabel('Pierwsza składowa główna')
plt.ylabel('Druga składowa główna')

plt.subplot(1, 3, 3)
comparison = (kmeans_labels == ms_labels).astype(int)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=comparison, cmap='RdYlBu', alpha=0.6)
plt.title('Porównanie metod (żółty=zgodne, niebieski=różne)')
plt.xlabel('Pierwsza składowa główna')
plt.ylabel('Druga składowa główna')

plt.tight_layout()
plt.show()

agreement = np.mean(kmeans_labels == ms_labels)
print(f"Zgodność między K-means i Mean Shift: {agreement:.2%}")

print("\n=== 5. Gaussowe modele mieszanki (GMM) ===")

gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(data_scaled)

data['gmm_cluster'] = gmm_labels

gmm_silhouette = silhouette_score(data_scaled, gmm_labels)
gmm_calinski = calinski_harabasz_score(data_scaled, gmm_labels)
gmm_davies = davies_bouldin_score(data_scaled, gmm_labels)

print(f"GMM - Liczba klastrów: 3")
print(f"Silhouette Score: {gmm_silhouette:.4f}")
print(f"Calinski-Harabasz Score: {gmm_calinski:.4f}")
print(f"Davies-Bouldin Score: {gmm_davies:.4f}")
print(f"Log-likelihood: {gmm.score(data_scaled):.2f}")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=gmm_labels, cmap='viridis', alpha=0.6)
plt.title('Mieszane modele rozkładów Gaussa na danych Iris')
plt.xlabel('Pierwsza składowa główna')
plt.ylabel('Druga składowa główna')

proba = gmm.predict_proba(data_scaled)
plt.subplot(1, 2, 2)
for i in range(3):
    plt.scatter(data_pca[:, 0], data_pca[:, 1], 
               c=proba[:, i], cmap=f'Blues', alpha=0.6,
               label=f'Klaster {i}')
plt.title('Prawdopodobieństwa przynależności do klastrów')
plt.xlabel('Pierwsza składowa główna')
plt.ylabel('Druga składowa główna')
plt.legend()

plt.tight_layout()
plt.show()

print("\n=== 6. Algorytm DBSCAN ===")

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(data_scaled)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

data['dbscan_cluster'] = dbscan_labels

print(f"DBSCAN - Liczba klastrów: {n_clusters_dbscan}")
print(f"Liczba punktów uznanych za szum: {n_noise}")

if n_clusters_dbscan > 1:
    non_noise_mask = dbscan_labels != -1
    if np.sum(non_noise_mask) > 0:
        dbscan_silhouette = silhouette_score(data_scaled[non_noise_mask], 
                                           dbscan_labels[non_noise_mask])
        print(f"Silhouette Score (bez szumu): {dbscan_silhouette:.4f}")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
unique_labels = set(dbscan_labels)
colors = ['yellow' if k == -1 else plt.cm.Spectral(k / len(unique_labels)) 
          for k in unique_labels]

for k, col in zip(unique_labels, colors):
    if k == -1:
        class_member_mask = (dbscan_labels == k)
        xy = data_pca[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6, alpha=0.6, label='Szum')
    else:
        class_member_mask = (dbscan_labels == k)
        xy = data_pca[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6, alpha=0.6, 
                label=f'Klaster {k}')

plt.title(f'DBSCAN na danych Iris')
plt.xlabel('Pierwsza składowa główna')
plt.ylabel('Druga składowa główna')
plt.legend()

plt.subplot(1, 2, 2)
methods = ['K-means', 'Mean Shift', 'GMM', 'DBSCAN']
scores = [kmeans_silhouette, 
         ms_silhouette if n_clusters_ms > 1 else 0,
         gmm_silhouette, 
         dbscan_silhouette if n_clusters_dbscan > 1 else 0]

plt.bar(methods, scores, color=['blue', 'green', 'red', 'orange'])
plt.title('Porównanie wyników Silhouette Score')
plt.ylabel('Silhouette Score')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("\n=== 7. Klastrowanie hierarchiczne ===")

agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg_clustering.fit_predict(data_scaled)

data['hierarchical_cluster'] = agg_labels

agg_silhouette = silhouette_score(data_scaled, agg_labels)
agg_calinski = calinski_harabasz_score(data_scaled, agg_labels)
agg_davies = davies_bouldin_score(data_scaled, agg_labels)

print(f"Klastrowanie hierarchiczne - Liczba klastrów: 3")
print(f"Silhouette Score: {agg_silhouette:.4f}")
print(f"Calinski-Harabasz Score: {agg_calinski:.4f}")
print(f"Davies-Bouldin Score: {agg_davies:.4f}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sample_indices = np.random.choice(len(data_scaled), size=1000, replace=False)
sample_data = data_scaled[sample_indices]

linkage_matrix = linkage(sample_data, method='ward')
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title('Dendrogram hierarchicznego klastrowania Iris')
plt.xlabel('Indeks próbki lub (rozmiar klastra)')
plt.ylabel('Odległość')

plt.subplot(1, 2, 2)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=agg_labels, cmap='viridis', alpha=0.6)
plt.title('Klastrowanie aglomeracyjne (Agglomerative Clustering)')
plt.xlabel('Pierwsza składowa główna')
plt.ylabel('Druga składowa główna')

plt.tight_layout()
plt.show()

print("\n=== 8. Porównanie skuteczności metod grupowania ===")

results_summary = pd.DataFrame({
    'Metoda': ['K-means', 'Mean Shift', 'GMM', 'DBSCAN', 'Hierarchiczne'],
    'Liczba klastrów': [3, n_clusters_ms, 3, n_clusters_dbscan, 3],
    'Silhouette Score': [
        kmeans_silhouette,
        ms_silhouette if n_clusters_ms > 1 else np.nan,
        gmm_silhouette,
        dbscan_silhouette if n_clusters_dbscan > 1 else np.nan,
        agg_silhouette
    ],
    'Calinski-Harabasz Score': [
        kmeans_calinski,
        ms_calinski if n_clusters_ms > 1 else np.nan,
        gmm_calinski,
        np.nan,
        agg_calinski
    ],
    'Davies-Bouldin Score': [
        kmeans_davies,
        ms_davies if n_clusters_ms > 1 else np.nan,
        gmm_davies,
        np.nan,
        agg_davies
    ]
})

print("Podsumowanie wyników wszystkich metod:")
print(results_summary.to_string(index=False, float_format='{:.4f}'.format))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Porównanie różnych metod klastrowania', fontsize=16)

methods_data = [
    ('K-means', kmeans_labels),
    ('Mean Shift', ms_labels),
    ('GMM', gmm_labels),
    ('DBSCAN', dbscan_labels),
    ('Hierarchiczne', agg_labels)
]

for i, (method_name, labels) in enumerate(methods_data):
    row = i // 3
    col = i % 3
    
    axes[row, col].scatter(data_pca[:, 0], data_pca[:, 1], c=labels, 
                          cmap='viridis', alpha=0.6)
    axes[row, col].set_title(f'{method_name}')
    axes[row, col].set_xlabel('Pierwsza składowa główna')
    axes[row, col].set_ylabel('Druga składowa główna')

axes[1, 2].remove()

plt.tight_layout()
plt.show()

print("\n=== 9. Ocena i interpretacja wyników ===")

print("\nAnaliza charakterystyk klastrów (K-means):")
for cluster in range(3):
    cluster_data = data[data['kmeans_cluster'] == cluster]
    print(f"\nKlaster {cluster} (n={len(cluster_data)}):")
    print(f"  Średni limit kredytowy: {cluster_data['LIMIT_BAL'].mean():.0f}")
    print(f"  Średni wiek: {cluster_data['AGE'].mean():.1f}")
    print(f"  Odsetek defaultów: {cluster_data['default.payment.next.month'].mean():.2%}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for cluster in range(3):
    cluster_data = data[data['kmeans_cluster'] == cluster]['LIMIT_BAL']
    plt.hist(cluster_data, alpha=0.7, label=f'Klaster {cluster}', bins=20)
plt.xlabel('Limit kredytowy')
plt.ylabel('Częstość')
plt.title('Rozkład limitów kredytowych według klastrów')
plt.legend()

plt.subplot(1, 3, 2)
for cluster in range(3):
    cluster_data = data[data['kmeans_cluster'] == cluster]['AGE']
    plt.hist(cluster_data, alpha=0.7, label=f'Klaster {cluster}', bins=20)
plt.xlabel('Wiek')
plt.ylabel('Częstość')
plt.title('Rozkład wieku według klastrów')
plt.legend()

plt.subplot(1, 3, 3)
default_rates = [data[data['kmeans_cluster'] == i]['default.payment.next.month'].mean() 
                for i in range(3)]
plt.bar(range(3), default_rates, color=['blue', 'green', 'red'])
plt.xlabel('Klaster')
plt.ylabel('Odsetek defaultów')
plt.title('Odsetek defaultów według klastrów')
plt.xticks(range(3), [f'Klaster {i}' for i in range(3)])

plt.tight_layout()
plt.show()

print("\n=== Analiza zakończona ===")