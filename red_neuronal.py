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
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

input_size = input_data.size(1)  # Size of the input layer
hidden_size = 64
num_classes = len(set(etiquetas_convertidas))
criterion = nn.CrossEntropyLoss()
net = BirdClassificationNet(input_size, hidden_size, num_classes)
optimizer = optim.Adam(net.parameters(), lr=0.001)

batch_size = 4
dataset = torch.utils.data.TensorDataset(input_data, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_inputs, batch_labels in dataloader:
        optimizer.zero_grad()

        # Pass the sequences to the neural network
        outputs = net(batch_inputs)

        # Calculate the loss using the mask
        mask = torch.sum(batch_inputs != 0, dim=1) > 0  # Creating a mask to ignore the padded elements
        masked_outputs = outputs[mask]
        masked_labels = batch_labels[mask]
        loss = criterion(masked_outputs, masked_labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")

torch.save(net.state_dict(), 'modelo_entrenado.pth')

net = BirdClassificationNet(input_size, hidden_size, num_classes)
net.load_state_dict(torch.load('modelo_entrenado.pth'))
net.eval()

input_test = torch.tensor([1.2, 0.5, 0.8], dtype=torch.float32)  # Example test features
output_test = net(input_test.unsqueeze(0))  # Note the unsqueeze to add the batch dimension
predicted_label = torch.argmax(output_test).item()

predicted_label_text = list(etiquetas_numericas.keys())[list(etiquetas_numericas.values()).index(predicted_label)]
print("Predicted label:", predicted_label_text)