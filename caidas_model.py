import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

#cargamos datos
data = pd.read_csv("datos_acelerometro.csv")

X = data[["ax", "ay", "az", "gx", "gy", "gz"]].values
#vector de resultados
y = data["caida"].values

#media y transformaion estandar
scaler = StandardScaler()
#transformacion
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.save")

#separamos entrenamiento y pruebas, reservamos 20% para evaluacion
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#definimos modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#compilamos y entrenamos

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

#guardamos modelo
model.save("modelo_caidas.h5")

#convertimos para tensor flow lite micro
def representative_data_gen():
    for i in range(100):
        yield [X_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
open("modelo_caidas.tflite", "wb").write(tflite_model)

#para obtener parametro scale y cero_point

interpreter = tf.lite.Interpreter(model_path="modelo_caidas.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input quantization:", input_details[0]['quantization'])  # (scale, zero_point)
print("Output quantization:", output_details[0]['quantization'])

