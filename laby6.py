import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits, fetch_lfw_people, load_wine, fetch_20newsgroups
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import os

output_dir = "laby6_wykresy"

def zad1():
    """
    Redukcja wymiarowości za pomocą PCA dla Breast Cancer;
    """
    print("\n--- Zadanie 1: Breast Cancer ---")
    
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    
    print(f"Wymiar danych: {X.shape}")
    print(f"Liczba klas: {len(np.unique(y))}")
    print(f"Rozmiar klas: {np.bincount(y)}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled) # Redukcja wymiarowości za pomocą PCA
    
    print(f"Wariancja wyjaśniona przez pierwsze dwie składowe: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                  alpha=0.8, edgecolors='k', s=60)
    plt.colorbar(scatter)
    plt.title('Redukcja wymiarowości danych Breast Cancer za pomocą PCA')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.4f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.4f})')
    plt.grid(True)
    plt.legend(['Łagodny', 'Złośliwy'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "breast_cancer_pca.png"))
    plt.close()
    
    pca_full = PCA().fit(X_scaled)
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
    plt.xlabel('Liczba składowych')
    plt.ylabel('Skumulowana wariancja wyjaśniona')
    plt.title('Skumulowana wariancja wyjaśniona vs liczba składowych')
    plt.grid(True)
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% wariancji')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "breast_cancer_pca_cumulative.png"))
    plt.close()
    
    n_components_95 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1
    print(f"Liczba składowych potrzebnych do wyjaśnienia 95% wariancji: {n_components_95}")
    
    # return X_scaled, pca, X_pca, y

zad1()

def zad2():
    """
    Redukcja wymiarowości za pomocą t-SNE dla Digits
    """
    print("\n--- Zadanie 2: Digits ---")
    
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    print(f"Wymiar danych: {X.shape}")
    print(f"Liczba klas: {len(np.unique(y))}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    X_tsne = tsne.fit_transform(X_scaled) # Redukcja wymiarowości za pomocą t-SNE
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', 
                  alpha=0.8, edgecolors='k', s=60)
    plt.colorbar(scatter, ticks=range(10))
    plt.title('Redukcja wymiarowości danych Digits za pomocą t-SNE')
    plt.xlabel('t-SNE Wymiar 1')
    plt.ylabel('t-SNE Wymiar 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "digits_tsne.png"))
    plt.close()
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i], cmap='binary')
        ax.set_title(f'Cyfra: {digits.target[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "digits_examples.png"))
    plt.close()
    
    # return X_scaled, X_tsne, y

zad2()

def zad3():
    """
    Redukcja wymiarowości za pomocą NMF dla LFW
    """
    print("\n--- Zadanie 3: LFW ---")
    
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names

    print(f"Wymiar danych: {X.shape}")
    print(f"Liczba klas (osób): {len(np.unique(y))}")
    print(f"Nazwy klas: {target_names}")
    
    n_components = 15
    nmf = NMF(n_components=n_components, random_state=42, max_iter=300) # Redukcja wymiarowości za pomocą NMF
    X_nmf = nmf.fit_transform(X)
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i, ax in enumerate(axes.flat):
        if i < n_components:
            ax.imshow(nmf.components_[i].reshape(lfw_people.images[0].shape), 
                      cmap='viridis')
            ax.set_title(f'NMF Składowa {i+1}')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lfw_nmf_components.png"))
    plt.close()
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_nmf[:, 0], X_nmf[:, 1], c=y, cmap='plasma', 
                 alpha=0.8, edgecolors='k', s=60)
    plt.colorbar(scatter, ticks=range(len(target_names)), label='Osoba')
    plt.title('Redukcja wymiarowości danych LFW za pomocą NMF')
    plt.xlabel('NMF Składowa 1')
    plt.ylabel('NMF Składowa 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lfw_nmf_scatter.png"))
    plt.close()
    
    # Rekonstrukcja twarzy
    n_faces = 5
    fig, axes = plt.subplots(2, n_faces, figsize=(15, 6))
    
    for i in range(n_faces):
        axes[0, i].imshow(X[i].reshape(lfw_people.images[0].shape), cmap='gray')
        axes[0, i].set_title(f'Oryginał: {target_names[y[i]]}')
        axes[0, i].axis('off')
        
        reconst = np.dot(X_nmf[i], nmf.components_)
        axes[1, i].imshow(reconst.reshape(lfw_people.images[0].shape), cmap='gray')
        axes[1, i].set_title('Rekonstrukcja')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lfw_nmf_reconstruction.png"))
    plt.close()
    
    # return X, X_nmf, y, target_names

zad3()

def zad4():
    """
    Redukcja wymiarowości za pomocą SVD dla Wine
    """
    print("\n--- Zadanie 4: Wine ---")
    
    wine = load_wine()
    X = wine.data
    y = wine.target
    
    print(f"Wymiar danych: {X.shape}")
    print(f"Liczba klas: {len(np.unique(y))}")
    print(f"Nazwy cech: {wine.feature_names}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_svd = svd.fit_transform(X_scaled)     # Redukcja wymiarowości za pomocą SVD
    
    print(f"Wariancja wyjaśniona przez pierwsze dwie składowe: {np.sum(svd.explained_variance_ratio_):.4f}")
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_svd[:, 0], X_svd[:, 1], c=y, cmap='viridis', 
                  alpha=0.8, edgecolors='k', s=60)
    plt.colorbar(scatter, ticks=range(3), label='Klasa wina')
    plt.title('Redukcja wymiarowości danych Wine za pomocą SVD')
    plt.xlabel(f'SVD Składowa 1 ({svd.explained_variance_ratio_[0]:.4f})')
    plt.ylabel(f'SVD Składowa 2 ({svd.explained_variance_ratio_[1]:.4f})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wine_svd.png"))
    plt.close()
    
    svd_full = TruncatedSVD(n_components=X.shape[1]-1)
    svd_full.fit(X_scaled)
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(svd_full.explained_variance_ratio_), marker='o')
    plt.xlabel('Liczba składowych')
    plt.ylabel('Skumulowana wariancja wyjaśniona')
    plt.title('Skumulowana wariancja wyjaśniona vs liczba składowych')
    plt.grid(True)
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% wariancji')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wine_svd_cumulative.png"))
    plt.close()
    
    n_components_95 = np.argmax(np.cumsum(svd_full.explained_variance_ratio_) >= 0.95) + 1
    print(f"Optymalna liczba składowych (95% wariancji): {n_components_95}")
    
    # return X_scaled, X_svd, y

zad4()

def zad5():
    """
    Redukcja wymiarowości za pomocą LDA dla 20 Newsgroups
    """
    print("\n--- Zadanie 5: 20 Newsgroups ---")
    
    categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
    newsgroups = fetch_20newsgroups(subset='train', categories=categories,
                                   remove=('headers', 'footers', 'quotes'),
                                   random_state=42)
    
    print(f"Liczba dokumentów: {len(newsgroups.data)}")
    print(f"Liczba kategorii: {len(newsgroups.target_names)}")
    print(f"Kategorie: {newsgroups.target_names}")
    
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english') # Przetwarzanie tekstu za pomocą TF-IDF
    X_tfidf = tfidf_vectorizer.fit_transform(newsgroups.data)
    y = newsgroups.target
    
    print(f"Wymiar danych po wektoryzacji TF-IDF: {X_tfidf.shape}")
    
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X_tfidf.toarray(), y) # Redukcja wymiarowości za pomocą LDA
    
    plt.figure(figsize=(12, 10))
    colors = ['blue', 'green', 'red', 'purple']
    target_names = newsgroups.target_names
    
    for color, i, target_name in zip(colors, range(len(target_names)), target_names):
        plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], color=color, alpha=0.8,
                   label=target_name)
    
    plt.title('Redukcja wymiarowości danych 20 Newsgroups za pomocą LDA')
    plt.xlabel('LDA Składowa 1')
    plt.ylabel('LDA Składowa 2')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "newsgroups_lda.png"))
    plt.close()
    
    # return X_tfidf, X_lda, y, target_names

zad5()

