import os
import numpy as np
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import ITSCDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

Missing_value = 128.0
#torch.autograd.set_detect_anomaly(True)

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_D(scores, masks):
    n = scores.shape[0]
    loss = torch.sum((1-masks[:, 1:]) * torch.log((1-torch.sigmoid(scores)).clamp(min=1e-4)))
    loss += torch.sum(masks[:, 1:] * torch.log(torch.sigmoid(scores)))
    loss /= -n

    return loss

def loss_adv(scores, masks):
    n = scores.shape[0]
    return torch.sum((1-masks[:, 1:]) * torch.log((1-torch.sigmoid(scores)).clamp(min=1e-4)))/n

def loss_cls(logits, labels):
    return nn.functional.cross_entropy(logits, labels)

def loss_imp(completed_seq, imputation_seq, masks):
    completed_seq_tensor = torch.cat(completed_seq, dim=1)
    imputation_seq_tensor = torch.cat(imputation_seq, dim=1)
    n = completed_seq_tensor.shape[0]
    #print(completed_seq[0].shape)
    #print(completed_seq_tensor.shape)
    return torch.sum(
            torch.square(
             (completed_seq_tensor-imputation_seq_tensor)*masks[:, 1:, None]
            )
           )/n

'''
class Config(object):
    layer_num = 2
    hidden_size = 100
    learning_rate = 1e-3
    missing_frac = 0.2
    cell_type = 'GRU'
    lamda = 1
    D_epoch = 1
    GPU = '0'
    batch_size = None   #batch_size for train
    epoch = None    #epoch for train
    lamda_D = None  #epoch for training of Discriminator
    G_epoch = None  #epoch for training of Generator
    train_data_filename = None
    test_data_filename = None
'''

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.num_steps = config.num_steps
        self.input_dimension_size = config.input_dimension_size
        self.cell_type = config.cell_type
        self.lamda = config.lamda
        self.class_num = config.class_num
        self.layer_num = config.layer_num

        self.rnn_layers = [
            nn.GRUCell(input_size = self.input_dimension_size,
                       hidden_size = self.hidden_size).to(device) \
        ]
        self.rnn_layers.extend([
            nn.GRUCell(input_size = self.hidden_size,
                       hidden_size = self.hidden_size).to(device) \
            for _ in range(self.layer_num-1)
        ])

        self.imp_projection = nn.Linear(self.hidden_size, self.input_dimension_size)

    def forward(self, x, masks):
        # x is a batch of complete sequences, each of length num_steps
        # masks is a batch of masks for each sequence
        batch_size = x.shape[0]
        imputation_sequence = []
        completed_sequence = []

        hidden_state = self.init_hidden(batch_size).to(device)
        for time_step in range(self.num_steps):
            if time_step == 0:
                for layer in range(self.layer_num):
                    if layer == 0:
                        hidden_state = self.rnn_layers[layer](
                            x[:, 0, :], hidden_state
                        )
                    else:
                        hidden_state = self.rnn_layers[layer](
                            hidden_state, hidden_state
                        )
            else:
                for layer in range(self.layer_num):
                    if layer == 0:
                        hidden_state = self.rnn_layers[layer](
                            completed_sequence[time_step-1][:, 0, :],
                            hidden_state
                        )
                    else:
                        hidden_state = self.rnn_layers[layer](
                            hidden_state, hidden_state
                        )


            if time_step < self.num_steps - 1:
                imputation_sequence.append(
                    self.imp_projection(hidden_state).reshape(
                        batch_size, 1, self.input_dimension_size
                    )
                )
                completed_sequence.append(
                        masks[:, time_step:time_step+1, None] * x[:,time_step:time_step+1,:]+ \
                    (1-masks[:,time_step:time_step+1,None])*imputation_sequence[time_step]
                )
                next_input = completed_sequence[time_step]
        return hidden_state,completed_sequence, imputation_sequence

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

class Discriminator(nn.Module):
    def __init__(self,config):
        super(Discriminator, self).__init__()
        self.name = "Discriminator"
        self.fc1 = nn.Linear(
                (config.num_steps-1)*config.input_dimension_size,
                (config.num_steps-1)*config.input_dimension_size)
        self.fc2 = nn.Linear(
                (config.num_steps-1)*config.input_dimension_size,
                int(config.num_steps)//2*config.input_dimension_size)
        self.fc3 = nn.Linear(
                int(config.num_steps)//2*config.input_dimension_size,
                config.num_steps-1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size,int(config.hidden_size)//2)
        self.fc2 = nn.Linear(int(config.hidden_size)//2, 10)
        self.fc3 = nn.Linear(10, config.class_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def main(config):
    print (f'Training data: {config.train_data_filename}')
    train_dataset = ITSCDataset(config.train_data_filename, config.missing_frac)
    test_dataset = ITSCDataset(config.test_data_filename)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=0)

    config.class_num = train_dataset.num_classes
    config.num_steps = train_dataset[0][0].shape[0]
    config.input_dimension_size = train_dataset[0][0].shape[1]
    print(f'Num steps = {config.num_steps}')
    print(f'Input dimension size = {config.input_dimension_size}')
    print(f'Num classes = {config.class_num}')

    # ---------------train------------------
    G = Generator(config).to(device)
    C = Classifier(config).to(device)
    D = Discriminator(config).to(device)
    G_optim = optim.Adam(G.parameters(), lr=config.learning_rate)
    C_optim = optim.Adam(C.parameters(), lr=config.learning_rate*2)
    D_optim = optim.Adam(D.parameters(), lr=config.learning_rate)

    for epoch in range(config.epoch):
        epoch_loss_D = 0
        epoch_loss_ajrnn = 0
        epoch_loss_cls = 0
        epoch_loss_imp = 0
        epoch_loss_adv = 0
        samples = 0
        correct = 0
        n_batches = 0
        all_preds = []
        all_targets = []

        print(f'Epoch {epoch+1}/{config.epoch}: ')
        for batch_idx, (data,masks,targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            masks = masks.to(device=device)
            targets = targets.to(device=device)
            samples += data.shape[0]

            hidden_state, completed_seq, imputation_seq = G(data, masks)
            logits = C(hidden_state)

            completed_seq_tensor = torch.cat(completed_seq, dim=1)
            scores = D(completed_seq_tensor)

            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                correct += torch.sum((preds == targets).float()).detach().item()
                all_preds.append(preds)
                all_targets.append(targets)

            #loss calculations
            l_D = loss_D(scores,masks)

            #backward
            D_optim.zero_grad()
            l_D.backward(retain_graph=True)

            epoch_loss_D += l_D.detach().cpu().item()

            # Combined loss
            l_imp = loss_imp(completed_seq,imputation_seq,masks)
            l_cls = loss_cls(logits,targets)
            l_adv = loss_adv(scores,masks)
            l_ajrnn = l_cls + l_imp + config.lamda_D * l_adv
            epoch_loss_cls += l_cls.detach().cpu().item()
            epoch_loss_adv += l_adv.detach().cpu().item()
            epoch_loss_imp += l_imp.detach().cpu().item()
            epoch_loss_ajrnn += l_ajrnn.detach().cpu().item()

            C_optim.zero_grad()
            G_optim.zero_grad()

            for p in D.parameters(): p.requires_grad_(False)
            l_ajrnn.backward()
            for p in D.parameters(): p.requires_grad_(True)

            C_optim.step()
            G_optim.step()
            D_optim.step()

            n_batches += 1
        print(f'D Loss per batch: {epoch_loss_D/n_batches}')
        print(f'AJ-RNN Loss per batch: {epoch_loss_ajrnn/n_batches}')
        print(f'Classification loss per batch: {epoch_loss_cls/n_batches}')
        print(f'Imputation loss per batch: {epoch_loss_imp/n_batches}')
        print(f'Adversarial loss per batch: {epoch_loss_adv/n_batches}')
        print(f'Train Accuracy: {correct/samples*100:.2f}%')
    preds = torch.cat(all_preds, dim=0).detach().cpu().numpy()
    targets = torch.cat(all_targets, dim=0).detach().cpu().numpy()
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,required=True)
    parser.add_argument('--epoch',type=int,required=True)
    parser.add_argument('--lamda_D',type=float,required=True,help='coefficient that adjusts gradients propagated from discriminator')
    parser.add_argument('--missing_frac', type=float, required=False, default=0.2, help='Fraction of missing elements in sequence')
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
