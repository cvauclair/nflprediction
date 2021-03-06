import torch
import torch.utils.data
import pandas as pd
from datetime import datetime

import validation

# Training parameters
batch_size = 256
num_epochs = 256

# def random_model():
#     return NeuralNet(input_size, random.randint(output_size, input_size), output_size)

# Training function definition
def training_function(model, dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(batch_size))

    for n in range(0, num_epochs):
        for x, y in dataloader:
    		# Move tensors to device
            x = x.to(device)
            y = y.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            loss = criterion(model(x), y)

            # Backprop
            loss.backward()
            optimizer.step()

    return model

# Testing function definition
def testing_function(model, dataset):
    # Testing
    with torch.no_grad():
        x, y = dataset[:]

        # Move tensors to device
        x = x.to(device)
        y = y.to(device)

        y_prime = model(x)
        # print("Predicted:\n{}".format(y_prime))
        # print("Desired:\n{}".format(y))
        correct = (y == torch.argmax(y_prime, dim=1)).sum().double()
        return 1.0 - correct/float(y.size(0))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA is available: {}".format(torch.cuda.is_available()))
if torch.cuda.is_available():
	print("Using device: {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))

# Load data
data = pd.read_csv("data/processed.csv", index_col=0)

# Drop null columns
data.dropna(axis='columns', how='any', inplace=True)

# Isolate season 1980 data
# season_1980_data = data[data.index.str.contains('198009|198010|198011|198012|198101|198102')]
# data = data[~data.index.str.contains('198009|198010|198011|198012|198101|198102')]

# Training data
y_data = data[['win', 'tie', 'loss']]
x_data = data.drop(['win', 'tie', 'loss', 'team'], axis='columns').dropna(axis='columns', how='any')

# Data parameters 
input_size = len(x_data.columns)
output_size = len(y_data.columns)

# Create dataset
dataset = torch.utils.data.TensorDataset(torch.Tensor(x_data.values), torch.argmax(torch.Tensor(y_data.values), dim=1))

# Create model
hidden_size = 10
model = torch.nn.Sequential(
	torch.nn.Linear(input_size, hidden_size),
	torch.nn.Tanh(),
	torch.nn.Linear(hidden_size, output_size),
	torch.nn.Tanh()
).to(device)

# Set optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# CrossEntropyLoss weights
w = torch.Tensor([y_data['win'].sum(), y_data['tie'].sum(), y_data['loss'].sum()])
w = w/len(y_data.index)
criterion = torch.nn.CrossEntropyLoss(weight=w.to(device))

start_time = datetime.now()
loss = validation.holdout_validation(model=model, dataset=dataset, testing_ratio=0.1, training_function=training_function, testing_function=testing_function)
end_time = datetime.now()

print("Model (1HL size {}) accuracy: {:.4f} % ({})".format(hidden_size, 100 * (1.0-loss), end_time - start_time))