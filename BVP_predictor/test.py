import numpy as numpy
import torch
import argparse
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--data_path', default='/media/data/UBFG/DATASET_2/', type=str, help='Data path')
    parser.add_argument('--input_dim', default=60, type=int, help='Input dimension')
    parser.add_argument('--n_layers', default=1,  type=int, help='LSTM layers')
    parser.add_argument('--output_size', default=1, type=int, help='Output dimension')
    parser.add_argument('--hidden_dim_1', default=150, type=int, help='Hidden first dimension')
    parser.add_argument('--hidden_dim_2', default=100, type=int, help='Hidden second dimension')
    parser.add_argument('--hidden_dim_3', default=50,  type=int, help='Hidden third dimension')
    args = parser.parse_args()

    # Check for GPU or CPU
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        print('Loading to GPUs.')
        device = torch.device('cuda')
    else:
        print('Loading CPUs.')
        device = torch.device('cpu')

    # Define model
    hidden_dim = [args.hidden_dim_1, args.hidden_dim_2, args.hidden_dim_3]
    model = LSTM(args.input_dim, args.output_size, hidden_dim, args.n_layers)
    print('Defined a LSTM model.')

    # Load pre-trained model
    model.load_state_dict(torch.load(args.data_path+'output_training/trained_lstm.pt', map_location=device))

    # Load test set
    test_X = np.load(args.data_path+'output_training/test_X.npy')
    test_y = np.load(args.data_path+'output_training/test_y.npy')

    # Make prediction
    pred = model(torch.as_tensor(test_X, dtype=torch.float32))
    plt.figure(figsize=(80,7))
    plt.plot(test_y.reshape(-1)[:1000], label='truth')
    plt.plot(pred.detach().numpy().reshape(-1)[:1000], label='pred')
    plt.legend(prop={'size': 25})
    plt.savefig('pred.pdf')
    

