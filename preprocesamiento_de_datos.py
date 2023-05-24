import os
import librosa
import numpy as np
import torch
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

carpeta_audio = 'C:\\Users\\lenovo\\OneDrive\\Escritorio\\proyecto\\datos_de_entrenamiento'
archivos_audio = os.listdir(carpeta_audio)
n_fft = 1024
num_caracteristicas_seleccionadas = 3
etiquetas = ['Bulbul  naranjero', 'Candelita plomiza', 'Carbonero común', 'Cotorrita aliazul',
             'cucarachero pechihabano', 'cuclillo piquinegro', 'Fiofío gris del Atlántico',
             'mielero escamoso', 'Milano muslirrufo', 'Minero rojizo', 'Mirlo común', 'Pava amazónica',
             'Pijuí frentigrís', 'Tirahojas ogarití', 'Trompetero aliverde occidental', 'Tucán pechiblanco',
             'Urraca negra malaya', 'Zarcero páñido']

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
        
        # Aplicar aumentación de datos con Audiomentations
        augmented_audio = augmentations(samples=audio, sample_rate=sr)
        audio = augmented_audio
        
        stft = librosa.stft(audio, n_fft=n_fft)
        magnitud = np.abs(stft)
        mean = np.mean(magnitud)  # Calcula la media de las magnitudes
        std = np.std(magnitud)  # Calcula la desviación estándar de las magnitudes
        magnitud = (magnitud - mean) / std  # Normalización Z-score
        media_frecuencias = np.mean(magnitud, axis=1)
        
        correlaciones[etiqueta] = np.sort(media_frecuencias)[-num_caracteristicas_seleccionadas:]
        
    except Exception as e:
        print(f"Error al procesar el archivo {archivo}: {str(e)}")

input_data = torch.tensor(np.array(list(correlaciones.values())), dtype=torch.float32)

print("Dimensiones de input_data:", input_data.size())
