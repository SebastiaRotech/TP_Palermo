import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.callbacks import EarlyStopping

# Cargar el dataset
file_path = r"C:\Users\sorre\Desktop\Estudio\Universidad de Palermo\Proyecto Práctico\Dataset\extended_processed_dataset.csv"
dataset = pd.read_csv(file_path)

# Mostrar las primeras filas
print("Primeras filas del dataset:")
print(dataset.head())

# Revisar balance de clases
print("\nDistribución de clases:")
print(dataset['Etiqueta'].value_counts())

# Separar características (X) y etiquetas (y)
X = dataset['Irms_3'].values.reshape(-1, 1)  # Característica: Irms_3
y = dataset['Etiqueta'].values  # Etiqueta: Estado del sistema

# Codificar etiquetas en valores numéricos
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Guardar los datos de prueba
with open(r"C:\Users\sorre\Desktop\Estudio\Universidad de Palermo\Proyecto Práctico\Modelos\datos_prueba.pkl", "wb") as f:
    pickle.dump((X_test, y_test), f)
print("\nDatos de prueba guardados en 'datos_prueba.pkl' en la carpeta especificada.")

# Crear un modelo de red neuronal en TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,)),  # Entrada: una característica (Irms_3)
    tf.keras.layers.Dense(16, activation='relu'),  # Capa oculta con 16 neuronas
    tf.keras.layers.Dense(16, activation='relu'),  # Otra capa oculta
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Salida multiclase
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Configurar EarlyStopping para evitar sobreentrenamiento
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(X_train, y_train, 
                    epochs=50,  # Incrementado para mayor oportunidad de ajuste
                    validation_data=(X_test, y_test), 
                    callbacks=[early_stopping],
                    verbose=1)

# Evaluar el modelo en el conjunto de prueba
evaluation = model.evaluate(X_test, y_test, verbose=1)

# Mostrar los resultados de la evaluación y las clases
print("\nEvaluación del modelo:", evaluation)
print("Clases:", label_encoder.classes_)

# Gráficos de Precisión y Pérdida
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Graficar la precisión
plt.figure(figsize=(10, 6))
plt.plot(epochs, acc, label='Precisión de Entrenamiento')
plt.plot(epochs, val_acc, label='Precisión de Validación')
plt.title('Precisión durante el Entrenamiento y Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid()
plt.show()

# Graficar la pérdida
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, label='Pérdida de Entrenamiento')
plt.plot(epochs, val_loss, label='Pérdida de Validación')
plt.title('Pérdida durante el Entrenamiento y Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid()
plt.show()

# Guardar el modelo entrenado
output_path_h5 = r"C:\Users\sorre\Desktop\Estudio\Universidad de Palermo\Proyecto Práctico\Modelos\modelo_original.h5"
model.save(output_path_h5)
print(f"\nModelo entrenado guardado como '{output_path_h5}'.")

# Convertir el modelo a TensorFlow Lite sin cuantización
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo convertido
output_path_tflite = r"C:\Users\sorre\Desktop\Estudio\Universidad de Palermo\Proyecto Práctico\Modelos\modelo_tinyml.tflite"
with open(output_path_tflite, 'wb') as f:
    f.write(tflite_model)

print(f"Modelo TensorFlow Lite guardado correctamente en: {output_path_tflite}")
