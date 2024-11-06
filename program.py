import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import kagglehub

# path to downloaded dataset
path = "/Users/zachariahchristie/.cache/kagglehub/datasets/domnic/celsius-to-fahrenheit/versions/1/training.csv"
train_data = pd.read_csv(path)

#Extracting all rows from first column of train data for X, and all rows in last column for y
X_train = train_data.iloc[:,0].values
y_train = train_data.iloc[:,-1].values

#Reshapes both X_train and y_train to have one column and as many rows as needed
#Then scales each value independently in X_train and y_train to fit within 0 to 1
sc = MinMaxScaler()
sct = MinMaxScaler()
X_train = sc.fit_transform(X_train.reshape(-1,1))
y_train = sct.fit_transform(y_train.reshape(-1,1))

#Convert both arrays to float32, then convert into PyTorch rensors, finally reshape the tensors to have one column and as many rows as necessary
X_train = torch.from_numpy(X_train.astype(np.float32)).view(-1,1)
y_train = torch.from_numpy(y_train.astype(np.float32)).view(-1,1)

#Our input will be in celsius, our ouput fahrenheit
input_size = 1
output_size = 1

#Use nn.Linear to perform a linear transformation on our data to create our first linear layer
model = nn.Linear(input_size, output_size)

#Our loss function is mean squared error, we aim to reduce this loss using a stochastic gradient descent (SGD) 
learning_rate = 0.01
l = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

num_epochs = 2000

for epoch in range(num_epochs):
    #Forward feed
    y_pred = model(X_train.requires_grad_())

    #loss calculation
    loss = l(y_pred, y_train)

    #calculate gradients
    loss.backward()

    #update weights
    optimizer.step()

    #clear gradients from previous step
    optimizer.zero_grad()

    print('epoch {}, loss{}' .format(epoch, loss.item()))

#Detach gradients from the tensor
predicted = model(X_train).detach().numpy()

#Visualisation
plt.scatter(X_train.detach().numpy()[:100] , y_train.detach().numpy()[:100])
plt.plot(X_train.detach().numpy()[:100] , predicted[:100] , "red")
plt.xlabel("Celcius")
plt.ylabel("Fahrenhite")
plt.show()