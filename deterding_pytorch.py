#Two layer network for vowel classification - Deterding dataset.
from numpy import vstack
from pandas import read_csv
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import Tensor
from torch import eye
from torch import zeros
from torch import argmax
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        df = read_csv(path, header=None)
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        self.X = self.X.astype('float32')
        self.y = self.y.astype('int')
        self.y = encoder(self.y,11)
        
        
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
       
def encoder(labels, num_classes):
    y = eye(num_classes) 
    return y[labels] 

# model definition
class ONN(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(ONN, self).__init__()
        nhid=20
        self.hidden1 = Linear(n_inputs, nhid)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.activation1 = ReLU()

        self.hidden2 = Linear(nhid, 11)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.activation2 = Softmax(dim=1)
                
        
    # forward propagatation
    def forward(self, X):
      
        X = self.hidden1(X)
        X = self.activation1(X)
        
        X = self.hidden2(X)
        X = self.activation2(X)
       
        return X


def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    ds = DataLoader(dataset, batch_size=1, shuffle=True)
    
    return ds


def train_model(train_dl, model, epoch_max):
    
    lossfunc = BCELoss()

    BGP_trainer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    

    for epoch in range(epoch_max):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            
            BGP_trainer.zero_grad()
            
            yhat = model(inputs)
            
            loss = lossfunc(yhat, targets)
            
            loss.backward()
            
            BGP_trainer.step()
            
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()



def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        
        yhat = model(inputs)
        
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        
        yhat = yhat.round()
        
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc, predictions, actuals





# prepare the data
path = 'train.csv'
train_dl = prepare_data(path)

path = 'test.csv'
test_dl = prepare_data(path)

epoch_max = 100

print(len(train_dl.dataset), len(test_dl.dataset))

model = ONN(10)

train_model(train_dl, model, epoch_max)

acc, predictions, actuals = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)

#plotting confusion matrix

pclass=predictions.argmax(1)
tclass=actuals.argmax(1)
cm = confusion_matrix(tclass,pclass)
classes_labels = ('hid','hId','hEd','hAd','hYd','had','hOd','hod','hUd','hud','hed')
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, classes_labels)