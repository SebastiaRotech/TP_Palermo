import numpy as np
import tensorflow.lite as tflite
import pickle
import os

# Ruta del modelo cuantizado
dynamic_model_path = r"C:\Temp\Modelos\modelo_tinyml_dynamic.tflite"
int8_model_path = r"C:\Temp\Modelos\modelo_tinyml_int8.tflite"

# Ruta de los datos de prueba
test_data_path = r"C:\Temp\Modelos\datos_prueba.pkl"

# Verificar existencia de archivos
if not os.path.exists(dynamic_model_path):
    raise FileNotFoundError(f"No se encontró el modelo dinámico en: {dynamic_model_path}")

if not os.path.exists(int8_model_path):
    raise FileNotFoundError(f"No se encontró el modelo int8 en: {int8_model_path}")

if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"No se encontraron los datos de prueba en: {test_data_path}")

# Cargar los datos de prueba
with open(test_data_path, "rb") as f:
    X_test, y_test = pickle.load(f)
print("Datos de prueba cargados exitosamente.")

# Función para validar un modelo TensorFlow Lite
def validar_modelo_tflite(model_path, X_test, y_test):
    print(f"\nValidando modelo: {model_path}")
    # Cargar el modelo
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Obtener detalles de entrada y salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Normalizar las entradas si es necesario
    def preprocess_input(X):
        # Ajustar al tipo de datos esperado por el modelo
        expected_dtype = input_details[0]['dtype']
        return X.astype(expected_dtype)

    # Realizar predicciones en el conjunto de prueba
    tflite_predictions = []
    for sample in X_test:
        interpreter.set_tensor(input_details[0]['index'], preprocess_input(sample.reshape(1, -1)))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        tflite_predictions.append(np.argmax(output))

    # Calcular precisión del modelo
    tflite_accuracy = np.mean(np.array(tflite_predictions) == y_test)
    print(f"Precisión del modelo: {tflite_accuracy:.2%}")

# Validar modelo cuantizado dinámico
validar_modelo_tflite(dynamic_model_path, X_test, y_test)

# Validar modelo cuantizado int8
validar_modelo_tflite(int8_model_path, X_test, y_test)





























