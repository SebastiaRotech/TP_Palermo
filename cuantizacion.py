import tensorflow as tf
import os

# Ruta del modelo original
model_path = r"C:\Temp\Modelos\modelo_original.h5"
output_dir = r"C:\Temp\Modelos"

# Verificar existencia del modelo original
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No se encontró el modelo original en: {model_path}")

# Cargar el modelo entrenado
model = tf.keras.models.load_model(model_path)

# Cuantización dinámica
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_dynamic = converter.convert()
dynamic_path = os.path.join(output_dir, "modelo_tinyml_dynamic.tflite")
with open(dynamic_path, 'wb') as f:
    f.write(tflite_model_dynamic)
print(f"Modelo cuantizado dinámicamente guardado en: {dynamic_path}")

# Cuantización int8
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices([[0.24], [0.35], [4.5], [8.4]]).batch(1):
        yield [tf.cast(input_value, tf.float32)]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_types = [tf.int8]
tflite_model_int8 = converter.convert()
int8_path = os.path.join(output_dir, "modelo_tinyml_int8.tflite")
with open(int8_path, 'wb') as f:
    f.write(tflite_model_int8)
print(f"Modelo cuantizado int8 guardado en: {int8_path}")
