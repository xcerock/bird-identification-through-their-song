import os
import librosa
import numpy as np
import torch
import scipy.stats as stats
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

carpeta_audio = 'C:\\Users\\lenovo\\OneDrive\\Escritorio\\proyecto\\datos_de_entrenamiento'
archivos_audio = os.listdir(carpeta_audio)
n_fft = 1024
n_mels = 128
etiquetas = ['Bulbul naranjero', 'Candelita plomiza', 'Carbonero común', 'Cotorrita aliazul',
             'cucarachero pechihabano', 'cuclillo piquinegro', 'Fiofío gris del Atlántico',
             'mielero escamoso', 'Milano muslirrufo', 'Minero rojizo', 'Mirlo común', 'Pava amazónica',
             'Pijuí frentigrís', 'Tirahojas ogarití', 'Trompetero aliverde occidental', 'Tucán pechiblanco',
             'Urraca negra malaya', 'Zarcero páñido']


assert len(archivos_audio) == len(etiquetas), "Debe haber la misma cantidad de archivos de audio y etiquetas"

caracteristicas_seleccionadas = []
correlaciones = {}

# Definir transformaciones de aumentación de datos con Audiomentations
augmentations = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5)
])

for archivo, etiqueta in zip(archivos_audio, etiquetas):
    ruta_audio = os.path.join(carpeta_audio, archivo)

    try:
        audio, sr = librosa.load(ruta_audio)
        augmented_audio = augmentations(samples=audio, sample_rate=sr)

        mel_spectrogram = librosa.feature.melspectrogram(y=augmented_audio, sr=sr, n_fft=n_fft, n_mels=n_mels)


        mean = np.mean(mel_spectrogram)
        std = np.std(mel_spectrogram)
        median = np.median(mel_spectrogram)
        skewness = stats.skew(mel_spectrogram.flatten())
        kurtosis = stats.kurtosis(mel_spectrogram.flatten())

        features = np.array([mean, std, median, skewness, kurtosis])
        correlaciones[etiqueta] = features

    except Exception as e:
        print(f"Error al procesar el archivo {archivo}: {str(e)}")

input_data = torch.tensor(np.array(list(correlaciones.values())), dtype=torch.float32)
print("Dimensiones de input_data:", input_data.size())
