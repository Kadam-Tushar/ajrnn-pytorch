from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd 
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data import ITSCDataset
import torch
import numpy as np 
from ajrnn import Generator
# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ITSCDataset("dataset/CBF/CBF_TEST",0.2)
test_dataset = ITSCDataset("dataset/CBF/CBF_TEST")

train_loader = DataLoader(train_dataset, batch_size=20,shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=20,shuffle=True, num_workers=0)


G = torch.load("Gen.pt")
G = G.to(device)
hidden = []
target = []
for batch_idx, (data,masks,targets) in enumerate(tqdm(train_loader)):
    # Get data to cuda if possible
            data = data.to(device=device)
            masks = masks.to(device=device)
            targets = targets.to(device=device)
            target.append(targets)
            hidden_state, completed_seq, imputation_seq = G(data, masks)
            hidden.append(hidden_state)
            


# hidden = np.concatenate(hidden,axis = 0)
hidden = torch.cat(hidden,dim= 0).detach().cpu()
target = torch.cat(target,dim= 0).detach().cpu()
tsne_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(hidden)
pca_embedded = PCA(n_components=2).fit_transform(hidden)
df = pd.DataFrame(tsne_embedded,columns=['x-cord','y-cord'])
df['classes'] = target.numpy().tolist()
sns.scatterplot(data=df, x="x-cord", y="y-cord", hue="classes" , palette = sns.color_palette("tab10",n_colors=3)).set_title('TSNE plot')

plt.savefig("tsne_plots.png")

df = pd.DataFrame(pca_embedded,columns=['x-cord','y-cord'])
df['classes'] = target.numpy().tolist()

sns.scatterplot(data=df, x="x-cord", y="y-cord", hue="classes" ,  palette = sns.color_palette("tab10",n_colors=3) ).set_title('PCA plots')


plt.savefig("pca_plots.png")

