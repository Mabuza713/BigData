import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_environment():
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", keras.__version__)
    
    np.random.seed(42)
    tf.random.set_seed(42)

def load_and_explore_data():
    print("\n=== ZADANIE 1: Użycie biblioteki Keras ===")
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"Kształt danych treningowych: {x_train.shape}")
    print(f"Kształt etykiet treningowych: {y_train.shape}")
    print(f"Kształt danych testowych: {x_test.shape}")
    print(f"Kształt etykiet testowych: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def preprocess_fashion_mnist():
    print("\n=== ZADANIE 2: Fashion MNIST Dataset ===")
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    print(f"Fashion MNIST - dane treningowe: {x_train.shape}")
    print(f"Fashion MNIST - etykiety treningowe: {y_train.shape}")
    print(f"Klasy: {np.unique(y_train)}")
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    print(f"Nazwy klas: {class_names}")
    
    return (x_train, y_train), (x_test, y_test), class_names

def normalize_data(x_train, x_test):
    print("\n=== ZADANIE 3: Normalizacja danych ===")
    
    print(f"Przed normalizacją - min: {x_train.min()}, max: {x_train.max()}")
    
    x_train_norm = x_train.astype('float32') / 255.0
    x_test_norm = x_test.astype('float32') / 255.0
    
    print(f"Po normalizacji - min: {x_train_norm.min()}, max: {x_train_norm.max()}")
    
    return x_train_norm, x_test_norm

def create_neural_network_model(input_shape, num_classes):
    print("\n=== ZADANIE 4: Budowa modelu sieci neuronowej ===")
    
    model = Sequential([
        layers.Flatten(input_shape=input_shape),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    print("Architektura modelu:")
    model.summary()
    
    return model

def compile_and_train_model(model, x_train, y_train, x_test, y_test, epochs=10):
    print("\n=== ZADANIE 5: Kompilacja i trenowanie modelu ===")
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model skompilowany z:")
    print("- Optymalizator: Adam")
    print("- Funkcja straty: sparse_categorical_crossentropy")
    print("- Metryki: accuracy")
    
    print(f"\nRozpoczynanie trenowania na {epochs} epok...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=128,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    return history

def evaluate_model_performance(model, x_test, y_test, class_names):
    print("\n=== ZADANIE 6: Ocena wydajności modelu ===")
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Dokładność na zbiorze testowym: {test_accuracy:.4f}")
    print(f"Strata na zbiorze testowym: {test_loss:.4f}")
    
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, predicted_classes, target_names=class_names))
    
    cm = confusion_matrix(y_test, predicted_classes)
    
    return test_accuracy, test_loss, cm, predictions

def create_cnn_model(input_shape, num_classes):
    print("\n=== ZADANIE 7: Model CNN z warstwami konwolucyjnymi ===")
    
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    print("Architektura modelu CNN:")
    model.summary()
    
    return model

def visualize_results(history, cm, class_names):
    print("\n=== ZADANIE 8: Wizualizacja wyników i wnioski ===")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Dokładność treningowa')
    plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
    plt.title('Dokładność modelu')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Strata treningowa')
    plt.plot(history.history['val_loss'], label='Strata walidacyjna')
    plt.title('Strata modelu')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Macierz pomyłek')
    plt.xlabel('Predykcja')
    plt.ylabel('Rzeczywista klasa')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    prepare_environment()
    mnist_data = load_and_explore_data()
    
    (x_train, y_train), (x_test, y_test), class_names = preprocess_fashion_mnist()
    
    x_train_norm, x_test_norm = normalize_data(x_train, x_test)
    
    input_shape = (28, 28)
    num_classes = 10
    model = create_neural_network_model(input_shape, num_classes)
    
    history = compile_and_train_model(model, x_train_norm, y_train, 
                                    x_test_norm, y_test, epochs=10)
    
    accuracy, loss, cm, predictions = evaluate_model_performance(
        model, x_test_norm, y_test, class_names)
    
    print("\n=== BONUS: Model CNN ===")
    x_train_cnn = x_train_norm.reshape(-1, 28, 28, 1)
    x_test_cnn = x_test_norm.reshape(-1, 28, 28, 1)
    
    cnn_model = create_cnn_model((28, 28, 1), num_classes)
    cnn_model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
    
    cnn_history = cnn_model.fit(x_train_cnn, y_train,
                               epochs=5,
                               batch_size=128,
                               validation_data=(x_test_cnn, y_test),
                               verbose=1)
    
    cnn_accuracy, cnn_loss = cnn_model.evaluate(x_test_cnn, y_test, verbose=0)
    print(f"CNN - Dokładność: {cnn_accuracy:.4f}")
    
    visualize_results(history, cm, class_names)
    
    print(f"\n=== PODSUMOWANIE WYNIKÓW ===")
    print(f"Model podstawowy - Dokładność: {accuracy:.4f}")
    print(f"Model CNN - Dokładność: {cnn_accuracy:.4f}")
    print(f"Różnica: {cnn_accuracy - accuracy:.4f}")

if __name__ == "__main__":
    main()