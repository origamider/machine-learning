from module import *
import numpy as np
import ptb
from janome.tokenizer import Tokenizer

class RnnlmGen(RevisedRnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]
        
        x = start_id
        
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1,1)
            score = self.predict(x)
            p = softmax(score.flatten())
            sampled_id = np.random.choice(len(p),size=1,p=p)
            if (skip_ids is None) or (sampled_id not in skip_ids):
                x = sampled_id
                word_ids.append(int(x.item()))
        
        return word_ids

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

# 先頭50個の単語のidを出力
print(words[:50])
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RnnlmGen(vocab_size=vocab_size,wordvec_size=650,hidden_size=650)
model.load_params('lstm_wagahaiwa_nekodearu.pkl')
start_word = '私'
start_id = word_to_id[start_word]
print(f"start_id = {start_id}")
skip_ids = None
word_ids = model.generate(start_id,skip_ids)
text = ' '.join([id_to_word[i] for i in word_ids])
text = text.replace('。', '。\n')
print(text)
