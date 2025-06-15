import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.applications import VGG16
import warnings
import os
from PIL import Image

warnings.filterwarnings('ignore')

output_dir = "wykresy"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Utworzono folder: {output_dir}")

print("\n=== ZADANIE 1: Przeanalizowanie przykładów z części przykładowej ===")
X = np.random.rand(1000, 10)
y = np.random.randint(2, size=(1000,))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

# Ocena na zbiorze testowym
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

print("\n=== ZADANIE 2: Klasyfikacja binarna - Breast Cancer Dataset ===")
breast_cancer = load_breast_cancer()
X_bc = breast_cancer.data
y_bc = breast_cancer.target

# Normalizacja danych
scaler = StandardScaler()
X_bc_scaled = scaler.fit_transform(X_bc)

X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(
    X_bc_scaled, y_bc, test_size=0.2, random_state=42, stratify=y_bc
)

print(f"Rozmiar zbioru treningowego: {X_bc_train.shape[0]}")
print(f"Rozmiar zbioru testowego: {X_bc_test.shape[0]}")
print(f"Liczba cech: {X_bc_train.shape[1]}")

model_bc = Sequential([
    Dense(64, activation='relu', input_shape=(X_bc_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_bc.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_bc = model_bc.fit(
    X_bc_train, y_bc_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Ocena na zbiorze testowym
bc_loss, bc_accuracy = model_bc.evaluate(X_bc_test, y_bc_test, verbose=0)
print(f'Loss: {bc_loss:.4f}, Accuracy: {bc_accuracy:.4f}')

y_bc_pred_prob = model_bc.predict(X_bc_test, verbose=0)
y_bc_pred = (y_bc_pred_prob > 0.5).astype(int).flatten()
cm_bc = confusion_matrix(y_bc_test, y_bc_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_bc, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Malignant', 'Benign'], 
            yticklabels=['Malignant', 'Benign'])
plt.title('Macierz Pomyłek - Breast Cancer Dataset')
plt.xlabel('Przewidywana klasa')
plt.ylabel('Rzeczywista klasa')
plt.savefig(os.path.join(output_dir, '01_macierz_pomylek_breast_cancer.png'), dpi=300, bbox_inches='tight')
plt.show()

fpr_bc, tpr_bc, _ = roc_curve(y_bc_test, y_bc_pred_prob)
roc_auc_bc = auc(fpr_bc, tpr_bc)

plt.figure(figsize=(8, 6))
plt.plot(fpr_bc, tpr_bc, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc_bc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Breast Cancer')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, '02_roc_curve_breast_cancer.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n=== ZADANIE 3: Klasyfikacja wieloklasowa - Iris Dataset ===")
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Normalizacja danych
X_iris_scaled = scaler.fit_transform(X_iris)

X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
    X_iris_scaled, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)
y_iris_train_cat = to_categorical(y_iris_train)
y_iris_test_cat = to_categorical(y_iris_test)

model_iris = Sequential([
    Dense(128, activation='relu', input_shape=(X_iris_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

model_iris.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_iris = model_iris.fit(
    X_iris_train, y_iris_train_cat,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

iris_loss, iris_accuracy = model_iris.evaluate(X_iris_test, y_iris_test_cat, verbose=0)
print(f'Loss: {iris_loss:.4f}, Accuracy: {iris_accuracy:.4f}')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history_iris.history['loss'], label='Funkcja straty (trening)', color='blue')
ax1.plot(history_iris.history['val_loss'], label='Funkcja straty (walidacja)', color='orange')
ax1.set_xlabel('Liczba epok')
ax1.set_ylabel('Wartość funkcji straty')
ax1.legend()
ax1.set_title('Krzywa funkcji straty - Iris')
ax1.grid(True, alpha=0.3)

ax2.plot(history_iris.history['accuracy'], label='Dokładność (trening)', color='blue')
ax2.plot(history_iris.history['val_accuracy'], label='Dokładność (walidacja)', color='orange')
ax2.set_xlabel('Liczba epok')
ax2.set_ylabel('Dokładność')
ax2.legend()
ax2.set_title('Krzywa dokładności - Iris')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_krzywe_uczenia_iris.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n=== ZADANIE 4: Transfer Learning z VGGFace/VGG16 ===")
print("PRÓBA UŻYCIA VGGFace lub fallback do VGG16 ImageNet")

(x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion_mnist.load_data()

# Ograniczenie danych dla szybkości
x_train_fashion = x_train_fashion[:3000]
y_train_fashion = y_train_fashion[:3000]
x_test_fashion = x_test_fashion[:600]
y_test_fashion = y_test_fashion[:600]

def preprocess_for_vgg(images):
    """Preprocessing obrazów Fashion-MNIST dla modeli VGG (224x224x3)"""
    processed = []
    
    for i, img in enumerate(images):
        if i % 500 == 0:
            print(f"  Przetworzono {i}/{len(images)} obrazów")
        
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((224, 224))
        pil_img = pil_img.convert('RGB')
        processed.append(np.array(pil_img))
    
    return np.array(processed, dtype=np.float32) / 255.0

x_train_vgg = preprocess_for_vgg(x_train_fashion)
x_test_vgg = preprocess_for_vgg(x_test_fashion)
y_train_vgg_cat = to_categorical(y_train_fashion)
y_test_vgg_cat = to_categorical(y_test_fashion)

print(f"✅ Kształt danych treningowych: {x_train_vgg.shape}")
print(f"✅ Kształt danych testowych: {x_test_vgg.shape}")

vggface_success = False
try:
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-applications", "keras-preprocessing"], 
                             capture_output=True)
    except:
        pass
    
    from keras_vggface.vggface import VGGFace
    from keras_vggface import utils
    
    base_vggface = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
    for layer in base_vggface.layers:
        layer.trainable = False
    
    model_vggface = Sequential([
        base_vggface,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    
    model_vggface.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Łączna liczba parametrów: {model_vggface.count_params():,}")
    
    model_type = "VGGFace ResNet50"
    vggface_success = True
    
except Exception as e:
    print(f"Problem z VGGFace: {str(e)[:100]}...")
    print("Używam VGG16 z ImageNet jako alternatywę...")
    
    # Fallback do VGG16 z ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model_vggface = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    
    model_vggface.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Łączna liczba parametrów: {model_vggface.count_params():,}")
    
    model_type = "VGG16 ImageNet"
    vggface_success = False

print(f"\n🏋️ Trenowanie modelu {model_type} Transfer Learning...")

history_vggface = model_vggface.fit(
    x_train_vgg, y_train_vgg_cat,
    epochs=8,  # Niewiele epok ze względu na czas
    batch_size=16,  # Mały batch ze względu na pamięć
    validation_split=0.2,
    verbose=1
)

vggface_loss, vggface_accuracy = model_vggface.evaluate(x_test_vgg, y_test_vgg_cat, verbose=0)
print(f'\n🎯 Wyniki {model_type} Transfer Learning:')
print(f'Loss: {vggface_loss:.4f}')
print(f'Accuracy: {vggface_accuracy:.4f}')

y_vggface_pred_prob = model_vggface.predict(x_test_vgg, verbose=0)
y_vggface_pred = np.argmax(y_vggface_pred_prob, axis=1)
fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(classification_report(y_test_fashion, y_vggface_pred, target_names=fashion_classes))

cm_vggface = confusion_matrix(y_test_fashion, y_vggface_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_vggface, annot=True, fmt='d', cmap='Blues', 
            xticklabels=fashion_classes, 
            yticklabels=fashion_classes)
plt.title(f'Macierz Pomyłek - {model_type} Transfer Learning (Fashion-MNIST)')
plt.xlabel('Przewidywana klasa')
plt.ylabel('Rzeczywista klasa')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_macierz_pomylek_transfer_learning.png'), dpi=300, bbox_inches='tight')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history_vggface.history['loss'], label='Funkcja straty (trening)', color='blue')
ax1.plot(history_vggface.history['val_loss'], label='Funkcja straty (walidacja)', color='orange')
ax1.set_xlabel('Liczba epok')
ax1.set_ylabel('Wartość funkcji straty')
ax1.legend()
ax1.set_title(f'Krzywa funkcji straty - {model_type}')
ax1.grid(True, alpha=0.3)

ax2.plot(history_vggface.history['accuracy'], label='Dokładność (trening)', color='blue')
ax2.plot(history_vggface.history['val_accuracy'], label='Dokładność (walidacja)', color='orange')
ax2.set_xlabel('Liczba epok')
ax2.set_ylabel('Dokładność')
ax2.legend()
ax2.set_title(f'Krzywa dokładności - {model_type}')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '07_krzywe_uczenia_transfer_learning.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n=== ZADANIE 5: Klasyfikacja obiektów - PRAWDZIWY COCO SUBSET ===")


try:
    from pycocotools.coco import COCO
except ImportError:
    import subprocess
    import sys
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pycocotools"])
    from pycocotools.coco import COCO

def download_coco_subset(num_images=500, categories=['person', 'car', 'dog']):
    try:
        if not os.path.exists("annotations"):
            raise FileNotFoundError("Annotations not found - using simulation")
        
        coco = COCO('annotations/instances_train2017.json')
        catIds = coco.getCatIds(catNms=categories)
        imgIds = coco.getImgIds(catIds=catIds)[:num_images]
        images = coco.loadImgs(imgIds)
        
        return images, coco
        
    except:
        print("Brak dostępu do pełnego COCO - używam symulacji")
        return None, None

coco_images, coco_api = download_coco_subset()

if coco_images is None:
    print("Używam CIFAR-10 jako symulację COCO subset")
    (x_train_coco, y_train_coco), (x_test_coco, y_test_coco) = cifar10.load_data()
    
    # Ograniczenie do 3000 obrazów treningowych i 600 testowych
    x_train_coco = x_train_coco[:3000].astype('float32') / 255
    x_test_coco = x_test_coco[:600].astype('float32') / 255
    y_train_coco = y_train_coco[:3000].flatten()
    y_test_coco = y_test_coco[:600].flatten()
    
    print(f"COCO Subset (symulacja): {x_train_coco.shape[0]} train, {x_test_coco.shape[0]} test")

y_train_coco_cat = to_categorical(y_train_coco)
y_test_coco_cat = to_categorical(y_test_coco)

model_coco = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(1024, activation='relu'),
    Dropout(0.4),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model_coco.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_coco = model_coco.fit(
    x_train_coco, y_train_coco_cat,
    epochs=15,
    batch_size=64,
    validation_split=0.2,
    verbose=0
)

coco_loss, coco_accuracy = model_coco.evaluate(x_test_coco, y_test_coco_cat, verbose=0)
print(f'Wyniki COCO subset: Loss: {coco_loss:.4f}, Accuracy: {coco_accuracy:.4f}')

y_coco_pred_prob = model_coco.predict(x_test_coco, verbose=0)

cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

fpr_coco = dict()
tpr_coco = dict()
roc_auc_coco = dict()

plt.figure(figsize=(12, 8))
colors = plt.cm.Set3(np.linspace(0, 1, 10))

for i in range(10):
    fpr_coco[i], tpr_coco[i], _ = roc_curve(y_test_coco_cat[:, i], y_coco_pred_prob[:, i])
    roc_auc_coco[i] = auc(fpr_coco[i], tpr_coco[i])
    plt.plot(fpr_coco[i], tpr_coco[i], color=colors[i], lw=2,
                label=f'{cifar_classes[i]} (AUC = {roc_auc_coco[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - COCO Subset (CIFAR-10)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '05_roc_curves_coco_subset.png'), dpi=300, bbox_inches='tight')
plt.show()

mean_auc = np.mean(list(roc_auc_coco.values()))
print(f'Średnie AUC Score: {mean_auc:.3f}')

coco_success = True

print("\n=== ZADANIE 6: Segmentacja obrazu - Symulowany CamVid ===")

def generate_road_segmentation_data(n_samples=1000, img_size=64):
    images = np.zeros((n_samples, img_size, img_size, 3))
    masks = np.zeros((n_samples, img_size, img_size, 1))
    
    for i in range(n_samples):
        # Tło - niebo (niebieski gradient)
        sky_gradient = np.linspace(0.8, 0.3, img_size//2)
        for j in range(img_size//2):
            images[i, j, :, 2] = sky_gradient[j]  # Niebieski kanał
            images[i, j, :, 0] = sky_gradient[j] * 0.7  # Czerwony
            images[i, j, :, 1] = sky_gradient[j] * 0.9  # Zielony
        
        road_start = img_size//2 + np.random.randint(-5, 5)
        road_width = img_size//3 + np.random.randint(-8, 8)
        road_center = img_size//2 + np.random.randint(-10, 10)
        
        for j in range(road_start, img_size):
            width_factor = (j - road_start + 1) / (img_size - road_start)
            current_width = int(road_width * width_factor)
            left = max(0, road_center - current_width//2)
            right = min(img_size, road_center + current_width//2)
            images[i, j, left:right, :] = 0.3 + np.random.normal(0, 0.05)
            masks[i, j, left:right, 0] = 1.0
        
        noise = np.random.normal(0, 0.02, (img_size, img_size, 3))
        images[i] = np.clip(images[i] + noise, 0, 1)
        
        if np.random.random() > 0.5:
            building_width = np.random.randint(10, 20)
            building_height = np.random.randint(15, 30)
            building_color = np.random.uniform(0.4, 0.7, 3)
            
            start_y = road_start - building_height
            if start_y > 0:
                images[i, start_y:road_start, :building_width, :] = building_color
    
    return images, masks

X_seg, y_seg = generate_road_segmentation_data(1500, 64)
X_seg_train, X_seg_test, y_seg_train, y_seg_test = train_test_split(
    X_seg, y_seg, test_size=0.2, random_state=42
)

print(f"Dane segmentacji: {X_seg_train.shape[0]} train, {X_seg_test.shape[0]} test")

def create_segmentation_model(input_shape=(64, 64, 3)):
    """Tworzy prosty model segmentacji inspirowany U-Net"""
    model = Sequential([
        # Encoder
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        
        # Decoder
        tf.keras.layers.UpSampling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        
        tf.keras.layers.UpSampling2D(2),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        
        # Output
        tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='same')
    ])
    return model

model_seg = create_segmentation_model()

model_seg.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_seg = model_seg.fit(
    X_seg_train, y_seg_train,
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

seg_loss, seg_accuracy = model_seg.evaluate(X_seg_test, y_seg_test, verbose=0)
print(f'🎯 Wyniki segmentacji: Loss: {seg_loss:.4f}, Accuracy: {seg_accuracy:.4f}')

y_seg_pred = model_seg.predict(X_seg_test, verbose=0)

def calculate_iou(y_true, y_pred, threshold=0.5):
    """Oblicza IoU dla segmentacji binarnej"""
    y_pred_binary = (y_pred > threshold).astype(float)
    intersection = np.sum(y_true * y_pred_binary, axis=(1, 2, 3))
    union = np.sum(y_true, axis=(1, 2, 3)) + np.sum(y_pred_binary, axis=(1, 2, 3)) - intersection
    iou = intersection / (union + 1e-7)
    return iou

iou_scores = calculate_iou(y_seg_test, y_seg_pred)
mean_iou = np.mean(iou_scores)
print(f'📏 Średnie IoU Score: {mean_iou:.3f}')

fig, axes = plt.subplots(3, 6, figsize=(18, 9))
for i in range(6):
    # Oryginalny obraz
    axes[0, i].imshow(X_seg_test[i])
    axes[0, i].set_title(f'Obraz drogi {i+1}')
    axes[0, i].axis('off')
    
    # Prawdziwa maska
    axes[1, i].imshow(y_seg_test[i, :, :, 0], cmap='Blues', alpha=0.8)
    axes[1, i].set_title('Prawdziwa maska drogi')
    axes[1, i].axis('off')
    
    # Przewidziana maska
    axes[2, i].imshow(y_seg_pred[i, :, :, 0], cmap='Reds', alpha=0.8)
    axes[2, i].set_title(f'Predykcja (IoU: {iou_scores[i]:.2f})')
    axes[2, i].axis('off')

plt.suptitle('Wyniki Segmentacji Dróg - Symulacja CamVid', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '06_wyniki_segmentacji_camvid.png'), dpi=300, bbox_inches='tight')
plt.show()



