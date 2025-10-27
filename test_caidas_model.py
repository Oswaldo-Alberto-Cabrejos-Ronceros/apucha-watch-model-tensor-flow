import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="modelo_caidas.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scale, zero_point = input_details[0]['quantization']

# Ejemplo: tomar una muestra del test (ya escalada)
x = X_test[0].astype(np.float32)

# Convertir float -> int8 según quant params
q_input = np.round(x / scale + zero_point).astype(np.int8)
interpreter.set_tensor(input_details[0]['index'], q_input.reshape(1, -1))
interpreter.invoke()
q_output = interpreter.get_tensor(output_details[0]['index'])
# Descuantizar (si necesitas la probabilidad original)
out_scale, out_zero_point = output_details[0]['quantization']
float_output = (q_output.astype(np.float32) - out_zero_point) * out_scale
print("Salida descuantizada:", float_output)