import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import japanize_matplotlib
from torch.utils.data import DataLoader,Dataset

N = 2000
x = np.linspace(-1,1,N)
y1 = np.sqrt(abs(x)) + np.sqrt(1-x**2)# ハートの上側
y2 = np.sqrt(abs(x)) - np.sqrt(1-x**2)# ハートの下側

# ハートのグラフ可視化
plt.plot(x,y1,color="red")
plt.plot(x,y2,color="red")
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.title("ハートの関数")
plt.show()

# データの用意
x = np.concatenate([x,x],axis=0)# xに対してy1、y2の2通りあるので、xを複製。
y = np.concatenate([y1,y2],axis=0)

# カスタムデータセット定義
class CumstomDataset(Dataset):
    def __init__(self):
        self.input = torch.tensor(x,dtype=torch.float32).unsqueeze(-1)# (size,)->(size,1) 最後の次元に1を挿入
        self.output = torch.tensor(y,dtype=torch.float32).unsqueeze(-1)
    
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self,idx):
        return self.input[idx],self.output[idx]

class MDN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim,hidden_dim)
        self.layer_mu = nn.Linear(hidden_dim,output_dim)
        self.layer_pi = nn.Linear(hidden_dim,output_dim)
        self.layer_sigma = nn.Linear(hidden_dim,output_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax()
    
    def forward(self,x):
        base = self.relu(self.layer1(x))
        mu = self.layer_mu(base)
        sigma = self.softplus(self.layer_sigma(base)) + 1e-2 #sigmaが小さくなりすぎないように、1e-2を足している。
        pi = self.softmax(self.layer_pi(base))
        return mu, sigma, pi

# 平均μ,標準偏差σ正規分布の確率密度関数。損失関数として使用するため、tensor型ね。(backpropagationするため)
def get_normal_distribution_prob(x,mu,sigma):
    return torch.exp((-(x - mu) ** 2) / (2 * sigma ** 2)) / ((2 * torch.pi) ** 0.5 * sigma)
    

def get_loss(inputs,labels,mu,sigma,pi):
    res = torch.tensor(0.0) #required_grad=Trueにするため、float型にしてね。
    batch_size = len(inputs)
    for i in range(batch_size):
        base = torch.tensor(0.0)#required_grad=Trueにするため、float型にしてね。
        for j in range(K):
            base += pi[i,j] * get_normal_distribution_prob(labels[i].item(),mu[i,j],sigma[i,j])
        res -= torch.log(base+1e-9) #log値が発散しないように
    
    res /= batch_size
    return res


#ハイパーパラメータ
input_dim = 1
hidden_dim = 128
output_dim = 1
batch_size = 40
K = 4
num_epochs = 100

dataset = CumstomDataset()
dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
model = MDN(input_dim,hidden_dim,K)
optimizer = optim.Adam(model.parameters())

# 学習
for epoch in range(num_epochs):
    loss_avg = 0
    ct = 0
    for inputs,labels in dataloader:
        mu, sigma, pi = model(inputs)
        optimizer.zero_grad()
        loss = get_loss(inputs,labels,mu,sigma,pi)
        loss.backward()
        optimizer.step()
        loss_avg += loss.item()
        ct += 1
    loss_avg /= ct
    if epoch % 10 == 0:
        print(f"loss = {loss_avg}")

test_inputs = torch.linspace(-1,1,1000).unsqueeze(-1) #(1000,)->(1000,1)
test_outputs = torch.zeros(1000)
with torch.no_grad():
    mu, sigma, pi = model(test_inputs)
    for i in range(1000):
        k = torch.distributions.Categorical(pi[i]).sample() #どの正規分布を選択するか
        m = torch.distributions.normal.Normal(mu[i,k],sigma[i,k]) #選択した正規分布でサンプリング
        test_outputs[i] = m.sample()

# 学習したハートの出力
plt.scatter(test_inputs.detach().numpy(),test_outputs.detach().numpy(),color="magenta")
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.title("学習後のハート関数")
plt.show()


with torch.no_grad():
    mu, sigma, pi = model(test_inputs)
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for k in range(K):
        axes[0].plot(test_inputs,mu[:,k])
    axes[0].set_xlim(-2,2)
    axes[0].set_ylim(-2,2)
    axes[0].set_title("各混合成分の平均 μ")
    
    for k in range(K):
        axes[1].plot(test_inputs,pi[:,k])
    axes[1].set_xlim(-1,1)
    axes[1].set_ylim(0,1)
    axes[1].set_title("各混合成分の選ばれる確率π")
    plt.show()