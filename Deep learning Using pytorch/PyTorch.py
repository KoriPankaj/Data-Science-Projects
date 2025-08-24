pip install torch torchvision

import torch

x = torch.Tensor(10).random_(0, 10)
x.to("cuda")
x.to("cpu")

torch.cuda.is_available()

tensor_1 = torch.tensor([1,1,0,2])
tensor_2 = torch.tensor([[0,0,2,1,2],[1,0,2,2,0]])

tensor_1.shape
tensor_2.shape
tensor = torch.tensor([1,1,0,2]).cuda()

example_1 = torch.randn(3,3)
example_2 = torch.randint(low=0, high=2, size=(3,3)).type(torch.FloatTensor)

tensor_1 = torch.tensor([0.1,1,0.9,0.7,0.3])
tensor_2 = torch.tensor([[0,0.2,0.4,0.6],[1,0.8,0.6,0.4]])
tensor_3 = torch.tensor([[[0.3,0.6],[1,0]], [[0.3,0.6],[0,1]]])

tensor_1 = torch.tensor([0.1,1,0.9,0.7,0.3]).cuda()
tensor_2 = torch.tensor([[0,0.2,0.4,0.6], [1,0.8,0.6,0.4]]).cuda()
tensor_3 = torch.tensor([[[0.3,0.6],[1,0]], [[0.3,0.6],[0,1]]]).cuda()

print(tensor_1.shape)
print(tensor_2.shape)
print(tensor_3.shape)

torch.Size([5])
torch.Size([2, 4])
torch.Size([2, 2, 2])

a = torch.tensor([5.0, 3.0], requires_grad=True)
b = torch.tensor([1.0, 4.0])
ab = ((a + b) ** 2).sum()
ab.backward()

print(a.grad.data)
print(b.grad.data)


import pandas as pd
data = pd.read_csv("C:/Users/Bharani Kumar/Downloads/energydata_complete.csv")
data = data.drop(columns=["date"])
data.head()
cols = data.columns
num_cols = data._get_numeric_data().columns
list(set(cols) - set(num_cols))
data.isnull().sum()
outliers = {}
for i in range(data.shape[1]):
    min_t = data[data.columns[i]].mean() - (3 * data[data.columns[i]].std())
    max_t = data[data.columns[i]].mean() + (3 * data[data.columns[i]].std())
    count = 0
    for j in data[data.columns[i]]:
        if j < min_t or j > max_t:
            count += 1
    percentage = count/data.shape[0]
    outliers[data.columns[i]] = "%.3f" % percentage

outliers

X = data.iloc[:,1:]
Y = data.iloc[:,0]

X = (X - X.min())/(X.max() - X.min())
X.head()

X.shape
train_end = int(len(X) * 0.6)
dev_end = int(len(X) * 0.8)

X_shuffle = X.sample(frac=1, random_state=0)
Y_shuffle = Y.sample(frac=1, random_state=0)

x_train = X_shuffle.iloc[:train_end,:]
y_train = Y_shuffle.iloc[:train_end]
x_dev = X_shuffle.iloc[train_end:dev_end,:]
y_dev = Y_shuffle.iloc[train_end:dev_end]
x_test = X_shuffle.iloc[dev_end:,:]
y_test = Y_shuffle.iloc[dev_end:]


print(x_train.shape, y_train.shape)
print(x_dev.shape, y_dev.shape)
print(x_test.shape, y_test.shape)

from sklearn.model_selection import train_test_split
x_new, x_test_2, y_new, y_test_2 = train_test_split(X_shuffle, Y_shuffle, test_size=0.2, random_state=0)
dev_per = x_test_2.shape[0]/x_new.shape[0]
x_train_2, x_dev_2, y_train_2, y_dev_2 = train_test_split(x_new, y_new, test_size=dev_per, random_state=0)

print(x_train_2.shape, y_train_2.shape)
print(x_dev_2.shape, y_dev_2.shape)
print(x_test_2.shape, y_test_2.shape)

import torch
import torch.nn as nn

x_train = torch.tensor(x_train.values).float()
y_train = torch.tensor(y_train.values).float()

x_dev = torch.tensor(x_dev.values).float()
y_dev = torch.tensor(y_dev.values).float()

x_test = torch.tensor(x_test.values).float()
y_test = torch.tensor(y_test.values).float()

model = nn.Sequential(nn.Linear(x_train.shape[1],100),
                      nn.ReLU(),
                      
                      nn.Linear(100,50),
                      nn.ReLU(),
                      
                      nn.Linear(50,25),
                      nn.ReLU(),
                     
                      nn.Linear(25,1))

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for i in range(1000):
    y_pred = model(x_train).squeeze()
    loss = loss_function(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i%100 == 0:
        print(i, loss.item())
        
pred = model(x_test[0])
print("Ground truth:", y_test[0].item(), "Prediction:",pred.item())
    