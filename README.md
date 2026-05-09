# machine-learning

> 公式ドキュメントや書籍を参考に、機械学習アルゴリズムを自分なりにわかりやすく実装して公開するリポジトリです。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

## ✨ Features

- 📖 **教科書・公式ドキュメント**を参考に、わかりやすい実装を目指しています
- 📦 **アルゴリズムごとに独立したディレクトリ**（必要なものだけ動かせる）
- 💬 **わかりにくい部分は日本語コメント**で補足

## 📚 実装一覧

| アルゴリズム | 種類 | 実装 | 主な参考文献 |
|------------|-----|-----|-------------|
| VAE (Variational Autoencoder) | 生成モデル | [`vae/`](./vae) | [Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114) / ゼロつく ❺ |

> 今後、他のアルゴリズム（GAN、Diffusion 等）も順次追加予定です。

## 🚀 Getting Started

### 1. リポジトリを取得

```bash
git clone https://github.com/origamider/machine-learning.git
cd machine-learning
```

### 2. 仮想環境の作成と有効化

```bash
python3 -m venv venv
source venv/bin/activate         # macOS / Linux
# venv\Scripts\activate          # Windows
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 4. 実行（VAE の例）

```bash
python vae/main.py
```

実行すると、MNIST の学習・潜在空間の可視化・格子状の生成画像が表示されます。

## 📁 ディレクトリ構成

```
machine-learning/
├── vae/                  # Variational Autoencoder
│   └── main.py
├── requirements.txt      # 依存パッケージ
├── LICENSE               # MIT License
└── README.md
```

## 🛠 動作環境

- Python 3.10 以上
- 主要パッケージ: PyTorch / torchvision / NumPy / matplotlib / japanize-matplotlib
- 詳細は [`requirements.txt`](./requirements.txt) を参照

## 🤝 Contributing

バグ報告・改善提案・実装の追加など、**Issue や Pull Request はお気軽にどうぞ**。

- 「ここの説明がわかりにくい」「このアルゴリズムも実装してほしい」といった要望も歓迎です

## 📄 License

本リポジトリは [MIT License](./LICENSE) のもとで公開しています。
商用・非商用問わず自由にご利用いただけます。

## 🙏 参考文献

- 斎藤康毅『[ゼロから作るDeep Learning ❺](https://github.com/oreilly-japan/deep-learning-from-scratch-5)』(O'Reilly Japan, 2024)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
