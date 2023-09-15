# Para capturar el fotograma
import cv2

# Para procesar la arreglo de la imagen
import numpy as np


# Importar el módulo tensorflow y cargar el modelo
import tensorflow as tf 


# Adjuntando el índice de la cámara como 0 con la aplicación del software
camera = cv2.VideoCapture(0)
mymodel= tf.keras.models.load_model("keras_model.h5")

# Bucle infinito
while True:

	# Leyendo/Solicitando un fotograma de la cámara
	status , frame = camera.read()

	# Si somos capaces de leer exitosamente el fotograma
	if status:

		# Voltear la imagen
		frame = cv2.flip(frame , 1)
		
		# Redimensionar el fotograma
		resized_frame=cv2.resize(frame,(224,224))
		# Expandir las dimensiones 
		resized_frame = np.expand_dims(resized_frame,axis=0)
		# Normalizar antes de alimentar al modelo
		resized_frame=resized_frame/255
		# Obtener predicciones del modelo
		predictions=mymodel.predict(resized_frame)
		rock=int(predictions[0][0]*100)
		paper=int(predictions[0][1]*100)
		scissor=int(predictions[0][2]*100)
		print(f"piedra: {rock}%,papel: {paper}%, tijeras: {scissor}%")
		
		
		# Mostrando los fotogramas capturados
		cv2.imshow('Alimentar' , frame)

		# Esperando 1ms
		code = cv2.waitKey(1)
		
		# Si se preciona la barra espaciadora, romper el bucle
		if code == 32:
			break

# Liberar la cámara de la aplicación del software
camera.release()

# Cerrar la ventana abierta
cv2.destroyAllWindows()
