import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from preprocesamiento_de_datos import *
from preprocesamiento_de_datos_prueba import *


# Convertir etiquetas de texto a valores numéricos
etiquetas_numericas = {
    'Bulbul naranjero': 0,
    'Candelita plomiza': 1,
    'Carbonero común': 2,
    'Cotorrita aliazul': 3,
    'cucarachero pechihabano': 4,
    'cuclillo piquinegro': 5,
    'Fiofío gris del Atlántico': 6,
    'mielero escamoso': 7,
    'Milano muslirrufo': 8,
    'Minero rojizo': 9,
    'Mirlo común': 10,
    'Pava amazónica': 11,
    'Pijuí frentigrís': 12,
    'Tirahojas ogarití': 13,
    'Trompetero aliverde occidental': 14,
    'Tucán pechiblanco': 15,
    'Urraca negra malaya': 16,
    'Zarcero páñido': 17
}

etiquetas_convertidas = [etiquetas_numericas[etiqueta] for etiqueta in etiquetas]

inputs = []  # Lista para almacenar las características seleccionadas

for etiqueta in etiquetas:
    caracteristicas_etiqueta = correlaciones.get(etiqueta, [])  # Obtener las características para la etiqueta

    if len(caracteristicas_etiqueta) > 0:
        inputs.append(caracteristicas_etiqueta)  # Agregar las características a la lista "inputs"

# Convertir la lista de inputs en tensores del mismo tamaño
max_length = max(len(x) for x in inputs)
inputs = [torch.tensor(x, dtype=torch.float32) for x in inputs]
inputs = [torch.cat((x, torch.zeros(max_length - len(x)))) for x in inputs]

# Convertir la lista de inputs en un tensor
input_data = pad_sequence(inputs, batch_first=True)
labels = torch.tensor(etiquetas_convertidas, dtype=torch.long)

# Verificar las dimensiones de input_data
print("Dimensiones de input_data:", input_data.size())

class BirdClassificationNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BirdClassificationNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# Parámetros de la red neuronal
input_size = input_data.size(1)  # Tamaño de la capa de entrada
hidden_size = 512  # Tamaño de las capas ocultas
num_classes = len(set(etiquetas_convertidas))  # Número de clases

criterion = nn.CrossEntropyLoss()

# Crear la instancia del modelo
modelo = BirdClassificationNet(input_size, hidden_size, num_classes)
optimizer = optim.Adam(modelo.parameters(), lr=0.0001)

batch_size = 4
dataset = TensorDataset(input_data, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Dividir los datos en entrenamiento y validación
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Crea dataloaders para los conjuntos de entrenamiento y validación
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Entrenamiento de la red neuronal
num_epochs = 20

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_inputs, batch_labels in train_dataloader:

        optimizer.zero_grad()

        # Pass the sequences to the neural network
        outputs = modelo(batch_inputs)

        # Calculate the loss using the mask
        mask = torch.sum(batch_inputs != 0, dim=1) > 0  # Creating a mask to ignore the padded elements
        masked_outputs = outputs[mask]
        masked_labels = batch_labels[mask]
        loss = criterion(masked_outputs, masked_labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Evaluación del rendimiento del modelo en el conjunto de validación
        val_running_loss = 0.0
        modelo.eval()
        for val_inputs, val_labels in val_dataloader:
            val_outputs = modelo(val_inputs)
            val_mask = (val_labels >= 0)
            val_loss = criterion(val_outputs[val_mask], val_labels[val_mask])
            val_running_loss += val_loss.item()
    
    val_epoch_loss = val_running_loss / len(val_dataloader)
    average_loss = running_loss / len(dataloader)
    print(f'Epoch: {epoch+1}, Train Loss: {average_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')


torch.save(modelo.state_dict(), 'modelo.pt')

# Verificar las dimensiones y valores de los datos de entrada
print("Dimensiones de input_data_pruebas:", input_data_pruebas.shape)
print("Valores de input_data_pruebas:", input_data_pruebas)

# Cargar el modelo entrenado

modelo = BirdClassificationNet(input_size, hidden_size, num_classes)
modelo.load_state_dict(torch.load('modelo.pt'))
modelo.eval()


# Realizar predicciones en los datos de prueba
outputs_test = modelo(input_data_pruebas)
predicted_labels = torch.argmax(outputs_test, dim=1)

# Convertir las etiquetas numéricas a etiquetas de texto
predicted_labels_text = [list(etiquetas_numericas.keys())[list(etiquetas_numericas.values()).index(label)] for label in predicted_labels]


# Imprimir las predicciones
print("Predicciones en los datos de prueba:")
for i, predicted_label_text in enumerate(predicted_labels_text):
    print(f"Dato de prueba {i+1}: {predicted_label_text}")
