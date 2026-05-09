'''
出典: 斎藤康毅, ゼロから作るDeep Learning ❺, オライリージャパン, 2024
'''
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import japanize_matplotlib

class Encoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,latent_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim,hidden_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.layer_mu = nn.Linear(hidden_dim,latent_dim)
        self.layer_sigma = nn.Linear(hidden_dim,latent_dim)
    
    def forward(self, x):
        base = self.relu(self.layer(x))
        mu = self.layer_mu(base)
        sigma = self.softplus(self.layer_sigma(base))
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self,latent_dim,hidden_dim,output_dim):
        super().__init__()
        self.l1 = nn.Linear(latent_dim,hidden_dim)
        self.l2 = nn.Linear(hidden_dim,output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,z):
        tmp = self.relu(self.l1(z))
        x_hat = self.sigmoid(self.l2(tmp))
        return x_hat

def create_z(mu, sigma):
    epsilon = torch.randn_like(sigma)
    z = mu + epsilon * sigma
    return z
    
class VAE(nn.Module):
    def __init__(self,input_dim,hidden_dim,latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim,hidden_dim,latent_dim)
        self.decoder = Decoder(latent_dim,hidden_dim,input_dim)
        self.mseloss = nn.MSELoss(reduction="sum")
    
    def get_loss(self, x):
        mu, sigma = self.encoder(x)
        z = create_z(mu, sigma)
        x_hat = self.decoder(z)
        return (self.mseloss(x_hat,x) * 0.5 - torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2) * 0.5) / len(x)

input_dim = 784
hidden_dim = 32
latent_dim = 2
batch_size = 64
num_epochs = 30
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(torch.flatten)
])
dataset = datasets.MNIST(
    root = "./data/",
    train = True,
    download= True,
    transform=transform
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)
model = VAE(input_dim,hidden_dim,latent_dim)
optimizer = optim.Adam(model.parameters())


# 学習
for epoch in range(num_epochs):
    loss_sum = 0
    ct = 0
    for x,label in dataloader:
        optimizer.zero_grad()
        loss = model.get_loss(x)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        ct += 1
    # lossの可視化
    loss_avg = loss_sum / ct
    print(loss_avg)

test_dataset = datasets.MNIST(
    root = "./data/",
    train = False,
    download= True,
    transform=transform
)
xs = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
labels = np.array([test_dataset[i][1] for i in range(len(test_dataset))])

mu, sigma = model.encoder(xs)
zs = create_z(mu, sigma)
zs = zs.detach().numpy()

# 0~9までの数字に該当する潜在変数zを出力する
for num in range(10):
    index = (labels == num)
    plt.scatter(zs[index,0],zs[index,1],label=str(num))
plt.legend()
plt.xlabel("z1")
plt.ylabel("z2")
plt.title("潜在変数zを2次元上で表示")
plt.show()

xs = torch.linspace(-3,3,20)
ys = torch.linspace(3,-3,20)
grid_y, grid_x = torch.meshgrid(ys,xs)
grid_x = torch.flatten(grid_x)
grid_y = torch.flatten(grid_y)
zs = torch.stack((grid_x,grid_y),dim=1)
x_hat = model.decoder(zs)
x_hat = x_hat.view(20,20,28,28).permute(0,2,1,3).reshape(560,560).detach().numpy()
plt.imshow(x_hat,cmap="gray")
plt.show()
