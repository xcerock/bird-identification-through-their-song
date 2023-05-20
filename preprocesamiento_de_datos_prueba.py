import os
import librosa
import numpy as np
import torch

carpeta_audio = 'C:\\Users\\lenovo\\OneDrive\\Escritorio\\proyecto\\datos_de_prueba'
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

for archivo, etiqueta in zip(archivos_audio, etiquetas):
    ruta_audio = os.path.join(carpeta_audio, archivo)

    try:
        audio, sr = librosa.load(ruta_audio)
        audio = librosa.util.normalize(audio)
        stft = librosa.stft(audio, n_fft=n_fft)
        magnitud = np.abs(stft)
        media_frecuencias = np.mean(magnitud, axis=1)

        std_media_frecuencias = np.std(media_frecuencias)

        if std_media_frecuencias != 0:
            if etiqueta not in correlaciones:
                correlaciones[etiqueta.strip()] = []

            correlaciones[etiqueta.strip()].extend(np.sort(media_frecuencias)[-num_caracteristicas_seleccionadas:])

    except Exception as e:
        print(f"Error al procesar el archivo {archivo}: {str(e)}")

etiquetas_numericas = {etiqueta: indice for indice, etiqueta in enumerate(set(etiquetas))}
etiquetas_convertidas = [etiquetas_numericas[etiqueta] for etiqueta in etiquetas]

etiquetas_convertidas = np.array(etiquetas_convertidas)

# Convertir a tensores de PyTorch
input_data = torch.tensor(list(correlaciones.values()), dtype=torch.float32)
targets = torch.tensor(etiquetas_convertidas, dtype=torch.long)

print("Dimensiones de input_data:", input_data.size())
print("Dimensiones de targets:", targets.size())
