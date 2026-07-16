# ref: ゼロから作るDeep Learning ❷
# ―自然言語処理編
# 斎藤 康毅　著
import numpy as np
import matplotlib.pyplot as plt
import ptb
from module import *
from janome.tokenizer import Tokenizer
import re

wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
max_epoch = 5
max_grad = 0.25
batch_size = 20
dropout = 0.5

def load_wagahaiwa_nekodearu():
    with open('wagahaiwa_nekodearu.txt', encoding='shift_jis') as f:
        text = f.read()
    text = re.split(r'-{20,}\n', text)[2]       # ヘッダ除去: 2本目の区切り線より後ろが本文
    text = re.split(r'底本：', text)[0]          # フッタ除去: 「底本：」より前が本文
    text = re.sub(r'※?［＃[^］]*］', '', text)   # 入力者注（orphanになる※も一括で）
    text = re.sub(r'《[^》]*》', '', text)        # ルビ
    text = text.replace('｜', '')                # ルビ開始記号（正規表現不要）
    text = re.sub(r'〔[^〕]*〕', '', text)        # アクセント分解された欧文
    text = re.sub(r'[ 　]', '', text)            # 空白
    text = re.sub(r'\n+', '\n', text).strip()    # 連続改行の圧縮
    return text

text = load_wagahaiwa_nekodearu()
t = Tokenizer()
words = list(t.tokenize(text, wakati=True))
word_to_id, id_to_word = {}, {}
for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

corpus = np.array([word_to_id[w] for w in words])
vocab_size = len(word_to_id)

xs = corpus[:-1]
ts = corpus[1:]

model = RevisedRnnlm(vocab_size, wordvec_size, hidden_size, dropout)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
trainer.plot()
model.save_params('lstm_wagahaiwa_nekodearu.pkl')