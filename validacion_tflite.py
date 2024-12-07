import os
import tensorflow.lite as tflite
import pickle
import numpy as np

# Rutas actualizadas
tflite_model_path = r"C:\Temp\Modelos\modelo_tinyml_regenerado.tflite"
datos_prueba_path = r"C:\Temp\Modelos\datos_prueba.pkl"

# Verificar existencia de los archivos
if not os.path.exists(tflite_model_path):
    print("El archivo .tflite no se encuentra en la ruta especificada.")
    exit()

if not os.path.exists(datos_prueba_path):
    print("El archivo datos_prueba.pkl no se encuentra en la ruta especificada.")
    exit()

# Cargar los datos de prueba
with open(datos_prueba_path, "rb") as f:
    X_test, y_test = pickle.load(f)
print("Datos de prueba cargados exitosamente.")

# Cargar el modelo TensorFlow Lite
try:
    interpreter = tflite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    print("Modelo TensorFlow Lite cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# Verificar detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"Detalles de entrada: {input_details}")
print(f"Detalles de salida: {output_details}")

# Normalizar las entradas si es necesario
def preprocess_input(X):
    return X.astype(np.float32)

# Realizar predicciones en el conjunto de prueba
tflite_predictions = []
for sample in X_test:
    interpreter.set_tensor(input_details[0]['index'], preprocess_input(np.array([sample])))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    tflite_predictions.append(np.argmax(output))

# Calcular precisión del modelo TensorFlow Lite
tflite_accuracy = np.mean(np.array(tflite_predictions) == y_test)
print(f"Precisión del modelo TensorFlow Lite: {tflite_accuracy:.2%}")










































































