import os
import numpy as np
import argparse
import torch 
from torch import nn
import torch.nn.functional as F 
from torch import optim 
from tqdm import tqdm
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.

Missing_value = 128.0

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def RNN_cell(type,input_size ,hidden_size ):
	if type == 'LSTM':
		cell = nn.LSTMCell(input_size,hidden_size)
	elif type == 'GRU':
		cell = nn.GRUCell(input_size, hidden_size)
	return cell

def load_data(filename):
	data_label = np.loadtxt(filename,delimiter=',')
	data = data_label[:,1:]
	label = data_label[:,0].astype(np.int32)
	return data, label

def transfer_labels(labels):
	#some labels are [1,2,4,11,13] and is transfer to standard label format [0,1,2,3,4]
	indexes = np.unique(labels)
	num_classes = indexes.shape[0]
	num_samples = labels.shape[0]

	for i in range(num_samples):
		new_label = np.argwhere( labels[i] == indexes )[0][0]
		labels[i] = new_label
	return labels, num_classes



class Config(object):
    layer_num = 1
    hidden_size = 100
    learning_rate = 1e-3
    cell_type = 'GRU'
    lamda = 1
    D_epoch = 1
    GPU = '0'
    '''User defined'''
    batch_size = None   #batch_size for train
    epoch = None    #epoch for train
    lamda_D = None  #epoch for training of Discriminator
    G_epoch = None  #epoch for training of Generator
    train_data_filename = None
    test_data_filename = None

def train_D(config,model,train_loader):
        model = model.to(device=device)
        # Loss and optimizer
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        # Train Network
        for epoch in range(config.D_epoch):
            for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
                # Get data to cuda if possible
                data = data.to(device=device)
                targets = targets.to(device=device)

                # Get to correct shape
                data = data.reshape(data.shape[0], -1)

                # forward
                scores = model(data)
                loss = criterion(scores, targets)

                # backward
                optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                optimizer.step()
        
def loss_D(scores,target):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(scores, target)
    return loss




    

class Discriminator(nn.Module):
    def __init__(self,config):
        super(Discriminator, self).__init__()
        self.name = "Discriminator"
        self.fc1 = nn.Linear(config.num_steps,config.num_steps)
        self.fc2 = nn.Linear(config.num_steps,int(config.num_steps)//2)
        self.fc3 = nn.Linear(int(config.num_steps)//2,config.num_steps)
        


    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x= F.tanh(self.fc2(x))
        predict_mask = self.fc3(x)
        return predict_mask

    
    


def main(config):
    print ('Loading data && Transform data--------------------')
    print (config.train_data_filename)
    train_data, train_label = load_data(config.train_data_filename)

    #For univariate
    config.num_steps = train_data.shape[1]
    config.input_dimension_size = 1

    train_label, num_classes = transfer_labels(train_label)
    config.class_num = num_classes

    print ('Train Label:', np.unique(train_label))
    print ('Train data completed-------------')

    test_data, test_labels = load_data(config.test_data_filename)

    test_label, test_classes = transfer_labels(test_labels)
    print ('Test data completed-------------')

    G = Generator(config)
    # G.build_model() remaining 
    real_pre = prediction * (1 - M) + prediction_target * M
    real_pre = torch.reshape(real_pre,[config.batch_size,(config.num_steps-1)*config.input_dimension_size])
    D = Discriminator(config)
    predict_M = D (real_pre)
    predict_M = torch.reshape(predict_M,[-1,(config.num_steps-1)*config.input_dimension_size])
    
#------------------------------------------------train---------------------------------------------------
    Epoch = config.epoch
    









if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size',type=int,required=True)
    parser.add_argument('--epoch',type=int,required=True)
    parser.add_argument('--lamda_D',type=float,required=True,help='coefficient that adjusts gradients propagated from discriminator')
    parser.add_argument('--G_epoch',type=int,required=True,help='frequency of updating AJRNN in an adversarial training epoch')
    parser.add_argument('--train_data_filename',type=str,required=True)
    parser.add_argument('--test_data_filename',type=str,required=True)

    parser.add_argument('--layer_num',type=int,required=False,default=1,help='number of layers of AJRNN')
    parser.add_argument('--hidden_size',type=int,required=False,default=100,help='number of hidden units of AJRNN')
    parser.add_argument('--learning_rate',type=float,required=False,default=1e-3)
    parser.add_argument('--cell_type',type=str,required=False,default='GRU',help='should be "GRU" or "LSTM" ')
    parser.add_argument('--lamda',type=float,required=False,default=1,help='coefficient that balances the prediction loss')
    parser.add_argument('--D_epoch',type=int,required=False,default=1,help='frequency of updating dicriminator in an adversarial training epoch')
    parser.add_argument('--GPU',type=str,required=False,default='0',help='GPU to use')

    config = parser.parse_args()
    main(config)