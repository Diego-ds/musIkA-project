import torch
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

#task = Task.init(project_name='MusIkA', task_name='EXP-1 MinMaxScaler')

parameters = {
    'n_layers': 2,
    'dropout_layers': [0.029809,0.000108966],
    'activation':'ELU',
    'optimizer':'Adam',
    'lr': 0.00615599,
    'bottleneck_size': 60,
    'loss':'MSE',
    'batch_size': 256
}

#parameters = task.connect(parameters)

# Convert the data into PyTorch tensors
train_tensor = torch.tensor(X_train)
test_tensor = torch.tensor(X_test)

# Define the dimensions of the input and latent space
input_dim = 193
latent_dim = 64

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 386),
            nn.ELU(),
            nn.Dropout(0.029809),
            nn.Linear(386, 48),
            nn.ELU(),
            nn.Dropout(0.000108966),
            nn.Linear(48, 60),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(60, 48),
            nn.ELU(),
            nn.Dropout(0.000108966),
            nn.Linear(48, 386),
            nn.ELU(),
            nn.Dropout(0.029809),
            nn.Linear(386, input_dim) 
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Create an instance of the autoencoder model
autoencoder = Autoencoder(input_dim, latent_dim).cuda()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.00615599)

# Train the autoencoder model
num_epochs = 100
batch_size = 256
parameters['epochs'] = num_epochs
train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
for epoch in range(num_epochs):
    running_loss = 0.0
    for data in train_loader:
        data = Variable(data).cuda()
        optimizer.zero_grad()
        recon_data = autoencoder(data.float())
        loss = criterion(recon_data, data.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_size
    epoch_loss = running_loss / len(X_train)
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))
    #Logger.current_logger().report_scalar("Loss graph", "train loss", iteration=epoch+1,value=epoch_loss)


# Test the autoencoder model  
    autoencoder.eval()
    with torch.no_grad():
        running_loss = 0.0
        for data in test_loader:
            data = Variable(data).cuda()
            recon_data = autoencoder(data.float())
            loss = criterion(recon_data, data.float())
            running_loss += loss.item() * batch_size
        test_loss = running_loss / len(X_test)
        print('Test Loss: {:.4f}'.format(test_loss))
        #Logger.current_logger().report_scalar("Loss graph", "test loss",iteration=epoch+1,value=test_loss)
    autoencoder.train()

# Encode the input data into the latent space
with torch.no_grad():
    test_tensor = Variable(test_tensor).cuda()
    encoded_data = autoencoder.encoder(test_tensor.float())
    print(encoded_data)