import torch
import torch.utils.data
import pandas as pd
from datetime import datetime

import validation

# NN definition
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size) 
        self.af1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)  
        self.af2 = torch.nn.Tanh()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.af1(out)
        out = self.fc2(out)
        out = self.af2(out)
        return out

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA is available: {}".format(torch.cuda.is_available()))
print("Using device: {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))

# Load data
x_data = pd.read_csv("data/x_reduced.csv")
y_data = pd.read_csv("data/y_reduced.csv")

# Data parameters 
input_size = len(x_data.columns)
output_size = len(y_data.columns)

dataset = torch.utils.data.TensorDataset(torch.Tensor(x_data.values), torch.argmax(torch.Tensor(y_data.values), dim=1))

model = NeuralNet(input_size, 10, output_size)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# CrossEntropyLoss weights
w = torch.Tensor([y_data['win'].sum(), y_data['tie'].sum(), y_data['loss'].sum()])
w = w/len(y_data.index)
criterion = torch.nn.CrossEntropyLoss(weight=w.to(device))

batch_size = 256
num_epochs = 256

def random_model():
    return NeuralNet(input_size, random.randint(output_size, input_size), output_size)

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

def testing_function(model, dataset):
    # Testing
    with torch.no_grad():
        x, y = dataset[:]

        # Move tensors to device
        x = x.to(device)
        y = y.to(device)

        y_prime = model(x)

        correct = (y == torch.argmax(y_prime, dim=1)).sum().double()
        return 1.0 - correct/float(y.size(0))

# model.apply(validation.init_weights)
# print(model.fc1.weight.data)
# m_prime = training_function(model, dataset)
# print(model.fc1.weight.data)
# print(m_prime.fc1.weight.data)

print("Evaluating model")
for n in range(8, 14):
	model = NeuralNet(input_size, n, output_size)
	model.to(device)

	start_time = datetime.now()
	# loss = validation.cross_validation(model=model, dataset=dataset, k=4, training_function=training_function, testing_function=testing_function)
	loss = validation.holdout_validation(model=model, dataset=dataset, testing_ratio=0.1, training_function=training_function, testing_function=testing_function)
	end_time = datetime.now()

	print("Model (1HL size {}) accuracy: {:.4f} % ({})".format(n, 100 * (1.0-loss), end_time - start_time))


# n = 9
# model = NeuralNet(input_size, n, output_size)
# model.to(device)

# start_time = datetime.now()
# loss = validation.holdout_validation(model=model, dataset=dataset, testing_ratio=0.1, training_function=training_function, testing_function=testing_function)
# end_time = datetime.now()

# print("Model (1HL size {}) accuracy: {:.4f} % ({})".format(n, 100 * (1.0-loss), end_time - start_time))