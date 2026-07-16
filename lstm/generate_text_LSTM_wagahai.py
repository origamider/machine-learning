from module import *
import numpy as np
import ptb
from janome.tokenizer import Tokenizer
import re

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

text  = load_wagahaiwa_nekodearu()
# print(text[15000:20000])
t = Tokenizer()
words = list(t.tokenize(text, wakati=True)) # 形態素解析
word_to_id, id_to_word = {}, {}
for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

corpus = np.array([word_to_id[w] for w in words])

# 先頭50個の単語のidを出力
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RnnlmGen(vocab_size=vocab_size,wordvec_size=650,hidden_size=650)
model.load_params('lstm_wagahaiwa_nekodearu.pkl') # trainLSTM.pyで学習した結果、作成されたpklを使用。
# ----------ここに最初の文字を入れてね。(文章に登場するやつ限定)(例えば、'私','真似','主人')--------------
start_word = '無法'
# -------------------------
start_id = word_to_id[start_word]
skip_ids = None
word_ids = model.generate(start_id,skip_ids)
text = ' '.join([id_to_word[i] for i in word_ids])
text = text.replace('。', '。\n')
print(text)
