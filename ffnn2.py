import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

initial_weights = torch.tensor([0.6, 0.2, 0.2, 0.1])
class AttentionLayer(nn.Module):
    def __init__(self, initial_weights):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(initial_weights, requires_grad=True)

    def forward(self, x):
        # Compute the dot product of the input and the attention weights
        attention_scores = torch.mul(x, self.attention_weights)
        # Apply softmax to get the attention distribution
        attention_distribution = F.softmax(attention_scores, dim=1)
        # Multiply the inputs by the attention distribution to get the output
        attended_data = torch.mul(x, attention_distribution)
        return attended_data


# Define the modified Actor class as the ThermalComfortModel
class ThermalComfortModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, initial_weights):
        super(ThermalComfortModel, self).__init__()
        self.attention = AttentionLayer(initial_weights)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.attention(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


# Create a custom dataset class
class ThermalComfortDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# Load cleaned dataset
df_cleaned = pd.read_csv('cleaned_asds32.csv')
# Split the dataset into input features (X) and target (y)
X = df_cleaned[['Air temperature', 'Relative humidity', 'Outdoor air temperature', 'Season']].values
y = df_cleaned['Thermal comfort'].astype(np.float32).values

# Split the dataset into training (60%), validation (20%), and testing (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=33) # Split 60% train, 40% temporary
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=33) # Split temporary set into 50% validation, 50% test

# Normalize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Create the custom dataset and data loader for training, validation, and testing sets
train_dataset = ThermalComfortDataset(X_train, y_train)
val_dataset = ThermalComfortDataset(X_val, y_val)
test_dataset = ThermalComfortDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



# Instantiate the model, loss function, and optimizer
input_size = 4
hidden_size = 128
output_size = 1
learning_rate = 0.0005

model = ThermalComfortModel(input_size, hidden_size, output_size, initial_weights)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
num_epochs = 100
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    train_loss = 0
    val_loss = 0
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = Variable(inputs), Variable(targets)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.item() * inputs.size(0)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
# plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Evaluate the model
model.eval()
with torch.no_grad():
    total_loss = 0
    total_samples = 0
    for inputs, targets in test_loader:
        inputs, targets = inputs, targets
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    mse = total_loss / total_samples
    print(f'Mean Squared Error on Test Set: {mse}')

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f'Linear Regression MSE: {mse_lr}')

# Support Vector Regression
svr_model = SVR(C=1e3, gamma=0.1)
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f'Support Vector Regression MSE: {mse_svr}')

# Decision Tree Regression
dt_model = DecisionTreeRegressor(random_state=40)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
print(f'Decision Tree Regression MSE: {mse_dt}')

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f'Random Forest Regression MSE: {mse_rf}')

# Neural Network MSE (already computed)
print(f'Neural Network MSE: {mse}')

# MSE values for all models
mse_values = [mse, mse_lr, mse_svr, mse_dt, mse_rf]
model_names = ['Our Neural Network', 'Linear Regression', 'SVM', 'Decision Tree', 'Random Forest']

# Create a bar plot
plt.bar(model_names, mse_values, width=0.3)
plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
plt.title('MSE Comparison of Regression Models')
plt.xticks(rotation=10)

# Show the plot
plt.show()
