import os
import librosa
import numpy as np
import torch
import scipy.stats as stats

carpeta_audio = 'C:\\Users\\lenovo\\OneDrive\\Escritorio\\proyecto\\datos_de_prueba'
archivos_audio = os.listdir(carpeta_audio)
n_fft = 1024
n_mels = 128

correlaciones_prueba = []

for archivo in archivos_audio:
    ruta_audio = os.path.join(carpeta_audio, archivo)

    try:
        audio, sr = librosa.load(ruta_audio)

        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, n_mels=n_mels)

        mean = np.mean(mel_spectrogram)
        std = np.std(mel_spectrogram)
        median = np.median(mel_spectrogram)
        skewness = stats.skew(mel_spectrogram.flatten())
        kurtosis = stats.kurtosis(mel_spectrogram.flatten())

        features_test = np.array([mean, std, median, skewness, kurtosis])
        correlaciones_prueba.append(features_test)

    except Exception as e:
        print(f"Error al procesar el archivo {archivo}: {str(e)}")

input_data_pruebas = torch.tensor(np.array(correlaciones_prueba), dtype=torch.float32)
print("Dimensiones de input_data_pruebas:", input_data_pruebas.size())
