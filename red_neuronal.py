import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from preprocesamiento_de_datos import *

# Convertir etiquetas de texto a valores numéricos
etiquetas_numericas = {
    'Bulbul  naranjero': 0,
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
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = pad_packed_sequence(x, batch_first=True)  # Desempaquetar la secuencia
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Crear una instancia de la red neuronal
input_size = input_data.size(1)  # Tamaño de la capa de entrada
hidden_size = 64
num_classes = len(set(etiquetas_convertidas))
criterion = nn.CrossEntropyLoss()


net = BirdClassificationNet(input_size, hidden_size, num_classes)

# Calcular las longitudes de las secuencias en el lote
lengths = [torch.sum(input_data[i, :] != 0) for i in range(input_data.size(0))]

# Definir la función de pérdida y el optimizador

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)



# Crear un DataLoader para manejar los datos de entrenamiento
batch_size = 4
dataset = torch.utils.data.TensorDataset(input_data, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Crear un DataLoader con shuffle=False
dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

# Entrenamiento de la red neuronal
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    
    for batch_inputs, batch_labels in dataloader:
        optimizer.zero_grad()
        
        # Calcular las longitudes de las secuencias en el lote
        lengths = [torch.sum(batch_inputs[i, :] != 0) for i in range(batch_inputs.size(0))]
        
        # Empaquetar las secuencias
        packed_inputs = pack_padded_sequence(batch_inputs, lengths, batch_first=True, enforce_sorted=False)
       
        # Pasar las secuencias empaquetadas a la red neuronal
        packed_outputs = net(packed_inputs)

        # Desempaquetar las secuencias
        outputs = packed_outputs.data

        # Calcular las longitudes de las secuencias en el lote
        lengths = torch.tensor([torch.sum(batch_inputs[i, :] != 0) for i in range(batch_inputs.size(0))])

        # Obtener la máscara de elementos válidos
        mask = torch.arange(outputs.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.expand_as(outputs)

 
        # Seleccionar los elementos válidos utilizando la máscara
        masked_outputs = torch.masked_select(outputs, mask)
        masked_labels = torch.masked_select(labels.repeat(1, max_length).unsqueeze(2).repeat(1, 1, 1), mask)

        # Calcular la pérdida utilizando los elementos seleccionados
        loss = criterion(masked_outputs, masked_labels)


        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    average_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")
    
# Guardar el modelo entrenado
torch.save(net.state_dict(), 'modelo_entrenado.pth')

# Cargar el modelo entrenado
net = BirdClassificationNet(input_size, hidden_size, num_classes)
net.load_state_dict(torch.load('modelo_entrenado.pth'))
net.eval()

# Realizar predicciones
input_test = torch.tensor([1.2, 0.5, 0.8], dtype=torch.float32)  # Ejemplo de características de prueba
output_test = net(input_test)
predicted_label = torch.argmax(output_test).item()

# Obtener la etiqueta predicha
etiqueta_predicha = list(etiquetas_numericas.keys())[list(etiquetas_numericas.values()).index(predicted_label)]
print("Etiqueta predicha:", etiqueta_predicha)
