import numpy as np
import matplotlib.pyplot as plt
import glob
import time
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import argparse

########################################################################################################################

def str2bool(v):
    # codes from: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

########################################################################################################################

class LSTM(nn.Module):
    def __init__(self, input_dim, output_size, hidden_dim, n_layers, drop_prob=None, drop_prob_lstm=None):

        super(LSTM, self).__init__()
        
        self.input_dim      = input_dim
        self.output_size    = output_size
        self.n_layers       = n_layers
        self.hidden_dim     = hidden_dim
        self.drop_prob      = drop_prob
        self.drop_prob_lstm = drop_prob_lstm

        self.lstm_1 = nn.LSTM(input_dim,     hidden_dim[0], n_layers, batch_first=True)  # dropout=drop_prob
        self.lstm_2 = nn.LSTM(hidden_dim[0], hidden_dim[1], n_layers, batch_first=True)  # dropout=drop_prob
        self.lstm_3 = nn.LSTM(hidden_dim[1], hidden_dim[2], n_layers, batch_first=False) # dropout=drop_prob

        if self.drop_prob is not None:
           print('Adding dropout to final dense layer')
           self.dropout = nn.Dropout(self.drop_prob)

        if self.drop_prob_lstm is not None:
           print('Adding spatial dropout after each LSTM layer')
           self.dropout_lstm = nn.Dropout2d(p=self.drop_prob_lstm)

        self.fc = nn.Linear(hidden_dim[2], output_size)

    def add_dropout_lstm(self, data):
        data = data.permute(0, 2, 1)   # convert to [batch, channels, time]
        data = self.dropout_lstm(data)
        data = data.permute(0, 2, 1)   # convert to [batch, time, channels]
        return data
        
    def forward(self, x): #, hidden):

        x, hidden = self.lstm_1(x) #, hidden)
        if self.drop_prob_lstm is not None:
           x = self.add_dropout_lstm(x)

        x, hidden = self.lstm_2(x) #, hidden)
        if self.drop_prob_lstm is not None:
           x = self.add_dropout_lstm(x)

        x, hidden = self.lstm_3(x) #, hidden)
        if self.drop_prob_lstm is not None:
           x = self.add_dropout_lstm(x)

        x = x[:, -1, :]

        if self.drop_prob is not None:
           x = self.dropout(x)

        x = self.fc(x)

        return x #, hidden

########################################################################################################################    

# Preprocessing:
# per video we choose 20 boxes
# each box gives us the intensity changes in 3 chavvels (RGB)
# 60 values per frame per subject

# Dividing in windows of size look_back:
# 60 windows of look_back points for each (frame - look_back)


def create_dataset(data_x, data_y, look_back=50):
    dataX, dataY = [], []
    for i in range(len(data_x)-look_back):
        a = data_x[i:(i+look_back)]
        dataX.append(a)
        dataY.append(data_y[i + look_back])
    return np.array(dataX), np.array(dataY)


def create_X_y(args):

    n_ok       = 0
    n_nosignal = 0
    n_except   = 0
        
    tab_X =[]
    tab_y =[]

    for subject_folder in sorted(glob.glob(args.data_path + 'subject*')):  

            print('Gathering data from subject', subject_folder)

            if os.path.exists(subject_folder+'/signals.npy'):
            
               try:
                  # Gathering video
                  data_X = np.load(subject_folder + '/signals.npy')
                  # Gathering BVP
                  data_y = np.loadtxt(subject_folder + '/ground_truth.txt')[0]
                        
                  X,y = create_dataset(data_X, data_y) 

                  if args.normalize_X:
                      min_   = np.min(X)
                      range_ = np.max(X) - min_ 
                      X = (X - min_) / range_  

                  if args.normalize_y:
                      min_   = np.min(y)
                      range_ = np.max(y) - min_ 
                      y = (y - min_) / range_ 

                  tab_X.append(X)
                  tab_y.append(y.reshape(-1,1))

                  n_ok += 1

               except Exception:
                  print('Error!')                  
                  n_except += 1

            else:
               print('File signals.npy does not exist')   
               n_nosignal += 1

    print(n_ok, ' successfully processed subjects')
    print(n_nosignal, ' subjects without signals.npy file')
    print(n_except, ' subjects with exceptions')

    # Use all subjects apart from last one for trainning dataset
    #train_X = np.concatenate(tab_X[:-1],axis=0)
    #train_y = np.concatenate(tab_y[:-1],axis=0)
    train_X = np.concatenate(tab_X[:-3],axis=0)
    train_y = np.concatenate(tab_y[:-3],axis=0)
    print('train_X.shape = ', train_X.shape)
    print('train_y.shape = ', train_y.shape)
    
    # Use last subject for testing dataset
    #test_X = tab_X[-1]
    #test_y = tab_y[-1]
    test_X = np.concatenate(tab_X[-3:], axis=0)
    test_y = np.concatenate(tab_y[-3:], axis=0)
    print('test_X.shape = ', test_X.shape)
    print('test_y.shape = ', test_y.shape)

    # Folder for saving trained model and test data sets for predictions
    if not os.path.exists(args.saving_path):
           os.makedirs(args.saving_path)
    
    if args.save_test:
       ## Save test dataset
       np.save(args.saving_path + 'test_X.npy', test_X)
       np.save(args.saving_path + 'test_y.npy', test_y)        

    return train_X, train_y, test_X, test_y

########################################################################################################################

def train(args, model, optimizer, scheduler, loss_fn, train_loader, val_loader, plot_losses_name=None):
    
    start = time.time()

    # Check for GPU
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        print('Training on GPUs')
        device = torch.device("cuda")
    else:
        print('Training on CPUs')
        device = torch.device("cpu")
        
    training_loss   = np.zeros(args.epochs)
    validation_loss = np.zeros(args.epochs)

    model.to(device)

    for epoch in range(args.epochs):

        model.train()
        
        for inputs, targets in train_loader:

            optimizer.zero_grad()
            inputs, labels = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            training_loss[epoch] += loss.data.item() * inputs.size(0)
            
        training_loss[epoch] /= len(train_loader.dataset)
        
        model.eval()
        
        for inputs, targets in val_loader:

            inputs, labels = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, labels) 
            validation_loss[epoch] += loss.data.item() * inputs.size(0)

        validation_loss[epoch] /= len(val_loader.dataset)        
            
        scheduler.step(validation_loss[epoch])

        print("Epoch: {}/{}   ".format(epoch+1, args.epochs),
              "Training Loss: {:.6f}   ".format(training_loss[epoch]),
              "Validation Loss: {:.6f}   ".format(validation_loss[epoch]))
        
    end = time.time()
    print('\nTraining took {} seconds.'.format(round(end - start, 2)))
    
    if plot_losses_name is not None:
        plt.plot(training_loss,   label='train loss')
        plt.plot(validation_loss, label='valid loss')
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.legend()
        title = 'hidden dims='+str(args.hidden_dim_1)+','+str(args.hidden_dim_2)+','+str(args.hidden_dim_3)+' drop_lstm='+str(args.drop_prob_lstm)+' loss='+args.loss_name
        plt.title(title, fontsize=8)
        plt.savefig(args.saving_path + plot_losses_name)
    
    torch.save(model,              args.saving_path + 'trained_lstm.model')
    torch.save(model.state_dict(), args.saving_path + 'trained_lstm.pt')




def plot_pred(args, data_X, data_y, model, pred_plot_name='pred.pdf'):

    print('Saving predictions plot')

    nsubplots = 3

    fig, axs = plt.subplots(nsubplots, figsize=(50, 12))

    title = 'hidden dims='+str(args.hidden_dim_1)+','+str(args.hidden_dim_2)+','+str(args.hidden_dim_3)+' drop_lstm='+str(args.drop_prob_lstm)+' loss='+args.loss_name
    fig.suptitle(title, fontsize=40)

    for i in range(nsubplots):

        subject_folder = sorted(glob.glob(args.data_path + 'subject*'))[-i-1]

        X_ = np.load(subject_folder + '/signals.npy')
        y_ = np.loadtxt(subject_folder + '/ground_truth.txt')[0]

        X,y = create_dataset(X_, y_)

        if args.normalize_X:
           min_   = np.min(X)
           range_ = np.max(X) - min_
           X = (X - min_) / range_

        if args.normalize_y:
           min_   = np.min(y)
           range_ = np.max(y) - min_
           y = (y - min_) / range_

        is_cuda = torch.cuda.is_available()
        if is_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        model.to(device)

        pred = model(torch.as_tensor(X, dtype=torch.float32).to(device))

        axs[i].plot(y.reshape(-1), label='truth')
        axs[i].plot(pred.cpu().detach().numpy().reshape(-1), label='pred')
        if i==0:
           axs[i].legend(prop={'size': 25})
        axs[i].set_title(subject_folder[subject_folder.find('subject'):], fontsize=30)

    plt.subplots_adjust(left=0.05, right=0.95, hspace=0.6)
    plt.savefig(pred_plot_name)

########################################################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--data_path',      default='/media/data/UBFG/DATASET_2/',  type=str,      help='Data path')
    parser.add_argument('--bvp_folder',     default='',                             type=str,      help='BVP folder name')
    parser.add_argument('--video_folder',   default='',                             type=str,      help='Video folder name')
    parser.add_argument('--batch_size',     default=72,                             type=int,      help='Batch size')
    parser.add_argument('--plot',           default='train_valid_loss.pdf',         type=str,      help='Name for losses plot')
    parser.add_argument('--epochs',         default=15,                             type=int,      help='Number of epochs')
    parser.add_argument('--input_dim',      default=60,                             type=int,      help='Input dimension')
    parser.add_argument('--n_layers',       default=1,                              type=int,      help='LSTM layers')
    parser.add_argument('--output_size',    default=1,                              type=int,      help='Output dimension')
    parser.add_argument('--hidden_dim_1',   default=60,                             type=int,      help='Hidden first dimension')
    parser.add_argument('--hidden_dim_2',   default=30,                             type=int,      help='Hidden second dimension')
    parser.add_argument('--hidden_dim_3',   default=10,                             type=int,      help='Hidden third dimension')
    parser.add_argument('--num_workers',    default=0,                              type=int,      help='Num_worker for data loader')
    #parser.add_argument('--pin_memory',    default=False,                          type=str2bool, help='Pin memory for data loader')
    parser.add_argument('--lr',             default=1e-4,                           type=float,    help='Optimiser learning rate')
    parser.add_argument('--normalize_X',    default=False,                          type=str2bool, help='Normalise X')
    parser.add_argument('--normalize_y',    default=False,                          type=str2bool, help='Normalise y')
    parser.add_argument('--save_test',      default=False,                          type=str2bool, help='Save X,y test dets for predictions')
    parser.add_argument('--drop_prob',      default=None,                           type=float,    help='Add dropout on dense layer with given probability')
    parser.add_argument('--drop_prob_lstm', default=0.6,                            type=float,    help='Add dropout on dense layer with given probability')
    parser.add_argument('--loss_name',      default='MSE',                          type=str,      help='Loss function choice')
    args = parser.parse_args()

    # Saving path
    args.saving_path = './output_training/out_epochs-'+str(args.epochs)+'_lr-'+str(args.lr)+'_hid1-'+str(args.hidden_dim_1)+'_hid2-'+str(args.hidden_dim_2)+'_hid3-'+str(args.hidden_dim_3)+'_nlayers-'+str(args.n_layers)+'_dropp-'+str(args.drop_prob)+'_dropp_lstm-'+str(args.drop_prob_lstm)+'_'+args.loss_name+'_normX-'+str(args.normalize_X)+'_normY-'+str(args.normalize_y)+'/'

    # Build data set
    train_X, train_y, test_X, test_y = create_X_y(args)

    # Build a train data loader
    train_data = TensorDataset(torch.as_tensor(train_X, dtype=torch.float32), torch.as_tensor(train_y, dtype=torch.float32))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    print('Built a train data loader')

    # Build test data loader
    test_data = TensorDataset(torch.as_tensor(test_X, dtype=torch.float32), torch.as_tensor(test_y, dtype=torch.float32))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=args.batch_size)
    print('Built a test data loader')

    # Define model
    hidden_dim = [args.hidden_dim_1, args.hidden_dim_2, args.hidden_dim_3]
    model = LSTM(args.input_dim, args.output_size, hidden_dim, args.n_layers, args.drop_prob, args.drop_prob_lstm)
    print('Defined a LSTM model')

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) #, betas=[.0, .5])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

    # Define loss function
    if args.loss_name=='MSE':
       loss_fn = nn.MSELoss()
    elif args.loss_name=='SmoothL1':
       loss_fn = nn.SmoothL1Loss()
    else:
       print('Loss function not recognised')

    # Train model
    train(args, model, optimizer, scheduler, loss_fn, train_loader, test_loader, args.plot)

    # Save prediction plot
    plot_pred(args, test_X, test_y, model, args.saving_path+'pred.pdf')
