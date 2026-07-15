import numpy as np
import matplotlib.pyplot as plt
import pickle

# 活性化関数やモデルの定義をする。パーツ部分を担当。

class BaseModel:
    def __init__(self):
        self.params, self.grads = None, None
    
    def forward(self, *args):
        return NotImplementedError
    
    def backward(self, *args):
        raise NotImplementedError
    
    def save_params(self, filename):
        params = [p.astype(np.float16) for p in self.params]
        with open(filename, 'wb') as f:
            pickle.dump(params, f)
    
    def load_params(self, filename):
        
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        params = [p.astype('f') for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]
    
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape
        
        val = np.dot(x, Wx) + np.dot(h_prev, Wh) + b
        
        f = val[:,:H]
        g = val[:,H:2*H]
        i = val[:,2*H:3*H]
        o = val[:,3*H:]
        
        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)
        
        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)
        
        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next
    
    def backward(self, dh_next, dc_next):
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache
        Wx, Wh, b = self.params
        tanh_c_next = np.tanh(c_next)
        ds = dc_next + dh_next * o * (1 - tanh_c_next ** 2)
        di = ds * g
        dg = ds * i
        dc_prev = ds * f
        do = dh_next * tanh_c_next
        df = ds * c_prev
        
        di = di * i * (1 - i)
        dg = dg * (1 - g ** 2)
        df = df * f * (1 - f)
        do = do * o * (1 - o)
        
        dval = np.hstack((df, dg, di, do))
        dWh = np.dot(h_prev.T, dval)
        dh_prev = np.dot(dval, Wh.T)
        dWx = np.dot(x.T,dval)
        dx = np.dot(dval,Wx.T)
        db = dval.sum(axis=0)
        
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        
        return dx, dh_prev, dc_prev
    
class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        
        self.h = None
        self.c = None
        self.layers = None
        self.stateful = stateful
    
    def forward(self,xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]
        hs = np.empty((N,T,H),dtype='f')
        self.layers = []  
        
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H),dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H),dtype='f')
            
            
        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:,t,:], self.h, self.c)
            hs[:,t,:] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        dh, dc = 0, 0
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]
        dxs = np.empty((N, T, D),dtype='f')
        
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c
    
    def reset_state(self):
        self.h, self.c = None, None


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.xs = None
    
    def forward(self, xs):
        N, T, D = xs.shape
        W, b = self.params #W.shapeは(D,H)だと思う
        rx = xs.reshape(N*T,-1)
        tmp = np.dot(rx, W) + b
        self.xs = xs
        return tmp.reshape(N,T,-1)
    
    def backward(self, dout):
        xs = self.xs
        N, T, D = xs.shape
        W, b = self.params
        dout = dout.reshape(N*T,-1)
        rx = xs.reshape(N*T,-1)
        db = np.sum(dout,axis=0)
        dW = np.dot(rx.T, dout)
        dxs = np.dot(dout,W.T)
        dxs = dxs.reshape(*xs.shape)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        
        return dxs

def softmax(x):
    if x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    elif x.ndim == 2:
        x = x - np.max(x,axis=1,keepdims=True)
        x = np.exp(x) / np.sum(np.exp(x),axis=1,keepdims=True)
    
    return x

class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1
    
    def forward(self, xs, ts):
        N, T, V = xs.shape
        
        if ts.ndim == 3:
            ts = ts.argmax(axis=2)
        
        mask = (ts != self.ignore_label)
        
        xs = xs.reshape(N*T,V)
        ts = ts.reshape(N*T)
        mask = mask.reshape(N*T)
        
        ys = softmax(xs)
        ls = np.log(ys[np.arange(N*T),ts])
        ls *= mask
        loss = -np.sum(ls)
        loss /= mask.sum()
        
        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache
        dx = ys #shape = (N*T,V)
        dx[np.arange(N*T),ts] -= 1 #イメージ: ys=(0.3,0.2,0.5),ts=(1,0,0)の時、(-0.7,0.2,0.5)みたいな。
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]
        dx = dx.reshape((N, T, V))
        return dx

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None

class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W
    
    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape
        out = np.empty((N, T, D), dtype="f")
        self.layers = []
        
        for t in range(T):
            layer = Embedding(self.W)
            out[:,t,:] = layer.forward(xs[:,t])
            self.layers.append(layer)
        
        return out
    
    def backward(self, dout):
        N, T, D = dout.shape
        
        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:,t,:])
            grad += layer.grads[0]
        
        self.grads[0][...] = grad
        return None

class SGD:
    '''
    確率的勾配降下法（Stochastic Gradient Descent）
    '''
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
            

class Rnnlm(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        embed_W = (rn(V,D) / 100).astype('f')
        lstm_Wx = (rn(D,4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H,4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(H,V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx,lstm_Wh,lstm_b,stateful=True),
            TimeAffine(affine_W,affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        self.lstm_layer.reset_state()

# 勾配爆発を防ぐためのclipping。
def clip_grads(grads, threshold):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    rate = threshold / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate
            
class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.current_epoch = 0
        self.current_idx = 0
        self.ppl_list = []
    
    def get_batch(self, xs, ts, batch_size, time_size):
        data_size = len(xs)
        jump = data_size // batch_size
        offsets = [i*jump for i in range(batch_size)]
        batch_x = np.empty((batch_size,time_size),dtype='i')
        batch_t = np.empty((batch_size,time_size),dtype='i')
        for i in range(time_size):
            for j in range(batch_size):
                batch_x[j , i] = xs[(offsets[j] + self.current_idx) % data_size]
                batch_t[j , i] = ts[(offsets[j] + self.current_idx) % data_size]
            self.current_idx += 1
        
        return batch_x, batch_t

    
    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35, max_grad=None, eval_interval=20):
        data_size = len(xs)
        max_iter = data_size // (batch_size*time_size)
        
        total_loss = 0
        loss_count = 0
        for epoch in range(max_epoch):
            for iter in range(max_iter):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                loss = self.model.forward(batch_x, batch_t)
                self.model.backward()
                if max_grad is not None:
                    clip_grads(self.model.grads, max_grad)
                self.optimizer.update(self.model.params,self.model.grads)
                total_loss += loss
                loss_count += 1
                
                if iter % eval_interval == 0:
                    ppl = np.exp(total_loss / loss_count)
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0
                    print(f"perplexity = {float(ppl)}")
    
    def plot(self):
        x = np.arange(len(self.ppl_list))
        plt.plot(x,self.ppl_list,label='train')
        plt.show()

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
    
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    
    def backward(self, dout):
        return dout * self.mask

class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = None
        
    def forward(self, xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs
    
    def backward(self, dout):
        return dout * self.mask

class RevisedRnnlm(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100, dropout_ratio=0.5):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        embed_W = (rn(V,D) / 100).astype('f')
        lstm_Wx1 = (rn(D,4*H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H,4*H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4*H).astype('f')
        lstm_Wx2 = (rn(H,4*H) / np.sqrt(H)).astype('f')
        lstm_Wh2 = (rn(H,4*H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4*H).astype('f')
        affine_b = np.zeros(V).astype('f')
        
        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(),
            TimeLSTM(lstm_Wx1,lstm_Wh1,lstm_b1,stateful=True),
            TimeDropout(),
            TimeLSTM(lstm_Wx2,lstm_Wh2,lstm_b2,stateful=True),
            TimeDropout(),
            TimeAffine(embed_W.T,affine_b) #D=Hを前提に、重み共有をしている
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def predict(self, xs, train_flg=False):
        for layer in self.drop_layers:
            layer.train_flg = train_flg
    
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        score = self.predict(xs,train_flg=True)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()