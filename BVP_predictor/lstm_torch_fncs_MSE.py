from imutils import face_utils
import sys
import numpy as np
import argparse
import imutils
import cv2
import scipy.signal as sgn
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot
from geomloss import SamplesLoss
import glob

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import argparse


class LSTM(nn.Module):
    def __init__(self, input_dim, output_size, hidden_dim, n_layers):

        super(LSTM, self).__init__()
        
        self.input_dim   = input_dim
        self.output_size = output_size
        self.n_layers    = n_layers
        self.hidden_dim  = hidden_dim
        
        self.lstm_1 = nn.LSTM(input_dim,     hidden_dim[0], n_layers, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden_dim[0], hidden_dim[1], n_layers, batch_first=True)
        self.lstm_3 = nn.LSTM(hidden_dim[1], hidden_dim[2], n_layers, batch_first=False)

        self.fc = nn.Linear(hidden_dim[2], output_size)
        
    def forward(self, x): #, hidden):
        lstm_out_1, hidden = self.lstm_1(x) #, hidden)
        lstm_out_2, hidden = self.lstm_2(lstm_out_1) #, hidden)
        lstm_out_3, hidden = self.lstm_3(lstm_out_2) #, hidden)
        last_frame_lstm_3 = lstm_out_3[:, -1, :]
        out = self.fc(last_frame_lstm_3)

        return out #, hidden
    
#    def init_hidden(self, batch_size):
#        weight = next(self.parameters()).data
#        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
#                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
#        return hidden


def create_dataset(data_x, data_y, look_back=50):
    dataX, dataY = [], []
    for i in range(len(data_x)-look_back-1):
        a = data_x[i:(i+look_back)]
        dataX.append(a)
        dataY.append(data_y[i + look_back])
    return np.array(dataX), np.array(dataY)


def create_X_y(data_path):
    # Find all subject folders
    folders = sorted(glob.glob(data_path+'subject*'))
    print('Folders found:\n', folders)
    # Construct X,y 
    tab_X =[]
    tab_Y =[]
    for subject in folders:
        print('Gathering data from ', subject)
        data_x = np.load(subject+'/signals.npy')
        data_y = np.loadtxt(subject+'/ground_truth.txt')[0].reshape(-1,1)
        X,Y = create_dataset(data_x, data_y) 
        tab_X.append(X)
        tab_Y.append(Y)
    # Use all subjects apart from last one for trainning dataset
    train_X = np.concatenate(tab_X[:-1],axis=0)
    train_y = np.concatenate(tab_Y[:-1],axis=0)
    print('train_X.shape = ', train_X.shape)
    print('train_y.shape = ', train_y.shape)
    # Use last subject for testing dataset
    test_X = tab_X[-1]
    test_y = tab_Y[-1]
    print('test_X.shape = ', test_X.shape)
    print('test_y.shape = ', test_y.shape)
    return train_X, train_y, test_X, test_y


def train(args):

    # Check for GPU
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        print('Training on GPUs.')
        device = torch.device("cuda")
    else:
        print('Training on CPUs.')
        device = torch.device("cpu")

    # Build X and y train and test sets and save test set for later analysis
    train_X, train_y, test_X, test_y = create_X_y(args.data_path)
    np.save(args.data_path+'test_X.npy', test_X)
    np.save(args.data_path+'test_y.npy', test_y)

    # Build a train data loader
    train_data = TensorDataset(torch.as_tensor(train_X, dtype=torch.float32), torch.as_tensor(train_y, dtype=torch.float32))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    print('Built a train data loader.')

    # Define model
    hidden_dim = [args.hidden_dim_1, args.hidden_dim_2, args.hidden_dim_3]
    model = LSTM(args.input_dim, args.output_size, hidden_dim, args.n_layers)
    print('Defined a LSTM model.')

    model.to(device)

    # Define Loss function
    lossfcn = nn.MSELoss()
    print('Defined loss function.')

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('Defined an Adam optimizer.')

    # Start training
    counter = 0
    model.train()
    print('Start training.')
    for i in range(args.epochs):
        print('i = ', i)
        for inputs, labels in train_loader:
            counter += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() # used to be model.zero_grad()
            output = model(inputs)
            loss = lossfcn(output, labels.float())
            loss.backward()
            optimizer.step()
            
            print("Epoch: {}/{}...".format(i+1, args.epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()))

    torch.save(model, args.data_path+'trained_lstm.model')
    torch.save(model.state_dict(), args.data_path+'trained_lstm.pt')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--data_path', default='/media/data/UBFG/DATASET_2/', type=str, help='Data path')
    parser.add_argument('--batch_size', default=72, type=int, help='Batch size')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--input_dim', default=60, type=int, help='Input dimension')
    parser.add_argument('--n_layers', default=1,  type=int, help='LSTM layers')
    parser.add_argument('--output_size', default=1, type=int, help='Output dimension')
    parser.add_argument('--hidden_dim_1', default=150, type=int, help='Hidden first dimension')
    parser.add_argument('--hidden_dim_2', default=100, type=int, help='Hidden second dimension')
    parser.add_argument('--hidden_dim_3', default=50,  type=int, help='Hidden third dimension')
    parser.add_argument('--num_workers', default=0, type=int, help='Num_worker for data loader')
    #parser.add_argument('--pin_memory', default=False, type=str2bool, help='Pin memory for data loader')
    parser.add_argument('--p', default=2,  type=int, help='Sinkhorn parameter')
    parser.add_argument('--blur', default=0.001,  type=int, help='Sinkhorn parameter')
    parser.add_argument('--lr', default=1e-3, type=float, help='Optimiser learning rate')
    args = parser.parse_args()

    train(args)

