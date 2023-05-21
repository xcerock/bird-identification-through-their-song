import os
import librosa
import numpy as np
import torch

carpeta_audio = 'C:\\Users\\lenovo\\OneDrive\\Escritorio\\proyecto\\datos_de_prueba'
archivos_audio = os.listdir(carpeta_audio)
n_fft = 1024
num_caracteristicas_seleccionadas = 3

correlaciones_prueba = []

for archivo in archivos_audio:
    ruta_audio = os.path.join(carpeta_audio, archivo)

    try:
        audio, sr = librosa.load(ruta_audio)
        stft = librosa.stft(audio, n_fft=n_fft)
        magnitud = np.abs(stft)
        media_frecuencias = np.mean(magnitud, axis=1)
        
        # Calcular la media y la desviación estándar de los datos de prueba
        mean_pruebas = np.mean(media_frecuencias)
        std_pruebas = np.std(media_frecuencias)
        
        # Aplicar la normalización Z-score a los datos de prueba
        media_frecuencias_prueba = (media_frecuencias - mean_pruebas) / std_pruebas

        std_media_frecuencias = np.std(media_frecuencias_prueba)

        if std_media_frecuencias != 0:
            correlaciones_prueba.append(np.sort(media_frecuencias_prueba)[-num_caracteristicas_seleccionadas:])

    except Exception as e:
        print(f"Error al procesar el archivo {archivo}: {str(e)}")

input_data_pruebas = torch.tensor(np.array(correlaciones_prueba), dtype=torch.float32)

print("Dimensiones de input_data_pruebas:", input_data_pruebas.size())
