# ref: ゼロから作るDeep Learning ❷
# ―自然言語処理編
# 斎藤 康毅　著
import numpy as np
import matplotlib.pyplot as plt
import ptb
from module import *
from janome.tokenizer import Tokenizer

wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
max_epoch = 5
max_grad = 0.25
batch_size = 20
dropout = 0.5

# corpus, word_to_id, id_to_word = ptb.load_data('train')
# corpus_test, _, _ = ptb.load_data('test')
with open('wagahaiwa_nekodearu.txt', encoding='shift_jis') as f:
    text = f.read()

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