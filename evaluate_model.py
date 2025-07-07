import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
MODEL_PATH = 'model/alexnet_final.keras'
STYLES_PATH = 'styles.csv'
IMAGES_PATH = 'images'  # ¡IMPORTANTE! Cambia esto si tus imágenes están en otra carpeta
IMG_SIZE = (227, 227)

# --- 1. Carga y Preparación de Datos ---
def load_and_prepare_data():
    print("Cargando y preparando datos...")
    df = pd.read_csv(STYLES_PATH, on_bad_lines='skip')
    df['image_path'] = df['id'].apply(lambda x: os.path.join(IMAGES_PATH, str(x) + '.jpg'))
    
    # Filtrar el DataFrame para mantener solo las filas donde el archivo de imagen realmente existe
    df = df[df['image_path'].apply(os.path.exists)]

    # Filtrar por las clases que el modelo conoce
    df = df[df['gender'].isin(['Men', 'Women'])]
    df = df[df['usage'].isin(['Casual', 'Sports'])]
    df = df.dropna(subset=['gender', 'usage'])

    # Codificar etiquetas
    gender_encoder = LabelEncoder().fit(df['gender'])
    usage_encoder = LabelEncoder().fit(df['usage'])
    df['gender_encoded'] = gender_encoder.transform(df['gender'])
    df['usage_encoded'] = usage_encoder.transform(df['usage'])

    # Dividir datos
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[['gender', 'usage']])
    print(f"Total de imágenes para evaluación: {len(test_df)}")
    return test_df, gender_encoder, usage_encoder

# --- 2. Generador de Datos para Evaluación ---
def image_generator(dataframe, batch_size=32):
    num_samples = len(dataframe)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_samples = dataframe.iloc[offset:offset+batch_size]
            
            images = []
            gender_labels = []
            usage_labels = []

            for _, row in batch_samples.iterrows():
                try:
                    img = tf.keras.preprocessing.image.load_img(row['image_path'], target_size=IMG_SIZE)
                    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                    images.append(img_array)
                    gender_labels.append(row['gender_encoded'])
                    usage_labels.append(row['usage_encoded'])
                except Exception as e:
                    # print(f"Error cargando imagen {row['image_path']}: {e}") # Descomentar para depurar
                    continue

            X = np.array(images)
            y_gender = np.array(gender_labels)
            y_usage = np.array(usage_labels)
            
            yield X, {"gender": y_gender, "usage": y_usage}

# --- 3. Evaluación del Modelo ---
def evaluate(model, test_df, gender_encoder, usage_encoder):
    print("\nIniciando evaluación del modelo...")
    test_gen = image_generator(test_df)
    steps = len(test_df) // 32

    y_pred_raw = model.predict(test_gen, steps=steps, verbose=1)
    
    # Extraer etiquetas verdaderas del generador
    y_true_gender, y_true_usage = [], []
    for i in range(steps):
        _, labels = next(image_generator(test_df, 32))
        y_true_gender.extend(labels['gender'])
        y_true_usage.extend(labels['usage'])

    # Decodificar predicciones
    # Asumiendo que la salida del modelo es una lista o un array [pred_gender, pred_usage]
    # Si es un solo array concatenado, hay que ajustarlo.
    # Basado en app.py, parece ser un solo array.
    pred_gender_indices = np.argmax(y_pred_raw[:, :2], axis=1)
    pred_usage_indices = np.argmax(y_pred_raw[:, 2:], axis=1)

    # Asegurarse que los arrays tengan el mismo tamaño
    min_len = min(len(pred_gender_indices), len(y_true_gender))
    pred_gender_indices = pred_gender_indices[:min_len]
    pred_usage_indices = pred_usage_indices[:min_len]
    y_true_gender = y_true_gender[:min_len]
    y_true_usage = y_true_usage[:min_len]

    # --- Métricas para GÉNERO ---
    print("\n" + "="*20 + " RESULTADOS DE GÉNERO " + "="*20)
    print(classification_report(y_true_gender, pred_gender_indices, target_names=gender_encoder.classes_))
    plot_confusion_matrix(y_true_gender, pred_gender_indices, gender_encoder.classes_, 'gender_confusion_matrix.png', 'Matriz de Confusión - Género')

    # --- Métricas para USO ---
    print("\n" + "="*20 + " RESULTADOS DE USO " + "="*20)
    print(classification_report(y_true_usage, pred_usage_indices, target_names=usage_encoder.classes_))
    plot_confusion_matrix(y_true_usage, pred_usage_indices, usage_encoder.classes_, 'usage_confusion_matrix.png', 'Matriz de Confusión - Uso')

# --- 4. Visualización ---
def plot_confusion_matrix(y_true, y_pred, classes, filename, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig(filename)
    print(f"\nMatriz de confusión guardada como: {filename}")

# --- Ejecución Principal ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en '{MODEL_PATH}'")
    elif not os.path.exists(STYLES_PATH):
        print(f"Error: No se encontró el archivo de estilos en '{STYLES_PATH}'")
    elif not os.path.exists(IMAGES_PATH):
        print(f"Error: No se encontró la carpeta de imágenes en '{IMAGES_PATH}'")
    else:
        model = tf.keras.models.load_model(MODEL_PATH)
        test_data, gender_enc, usage_enc = load_and_prepare_data()
        evaluate(model, test_data, gender_enc, usage_enc)
