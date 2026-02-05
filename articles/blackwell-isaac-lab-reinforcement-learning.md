---
title: "Blackwell実機検証：Isaac Labによる強化学習環境の構築"
emoji: "🧠"
type: "tech"
topics: ["IsaacLab", "強化学習", "Blackwell", "DGXSpark", "ロボティクス"]
published: true
---

[前回の記事](/articles/blackwell-isaac-sim-robot-learning)ではIsaac SimをDGX Spark上でDockerコンテナとして動作させ、合成データ生成までを検証しました。

今回は、強化学習フレームワーク**Isaac Lab**をソースからビルドし、実際にロボット学習タスクを実行するまでの手順をレポートします。

---

## エグゼクティブサマリー

| 項目 | 結果 |
|------|------|
| **Isaac Simソースビルド** | 成功（約8.5分） |
| **Isaac Labインストール** | 成功 |
| **Cartpole学習** | 成功（290K steps/s） |
| **ヘッドレス実行** | 問題なし |
| **GUI表示** | 現時点で非対応 |

---

## 1. Isaac Labとは

Isaac Labは、NVIDIA Isaac Sim上で動作する強化学習（Reinforcement Learning）フレームワークです。

| 特徴 | 説明 |
|-----|------|
| 大規模並列シミュレーション | 数千のロボットを同時に学習 |
| GPU最適化 | PhysXとPyTorchがGPU上で連携 |
| 豊富なタスク | Cartpole、Ant、Humanoid、産業用ロボット等 |
| モジュラー設計 | カスタムタスクの追加が容易 |

---

## 2. なぜソースビルドが必要か

DGX Sparkはaarch64（ARM64）アーキテクチャを採用しています。Isaac Sim 5.1.0のDockerイメージにはIsaac Labが同梱されておらず、またライブストリーミング機能もaarch64では非対応のため、**ソースからのビルド**が公式推奨の方法となっています。

---

## 3. 検証環境

| 項目 | 値 |
|------|-----|
| GPU | NVIDIA GB10（Blackwell） |
| Driver | 580.95.05 |
| CUDA | 13.0 |
| メモリ | 128GB UMA |
| OS | Ubuntu 24.04.2 LTS |
| アーキテクチャ | aarch64（ARM64） |
| GCC | 11.5.0（ビルド要件） |
| Isaac Sim | ソースビルド版 |
| Isaac Lab | ソースビルド版 |

---

## 4. セットアップ手順

### Step 1: 前提パッケージのインストール

Isaac Simのビルドには**GCC 11**が必要です。

```bash
# GCC 11とGit LFSのインストール
sudo apt update && sudo apt install -y gcc-11 g++-11 git-lfs

# GCC 11をデフォルトに設定
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200

# 確認
gcc --version
# gcc (Ubuntu 11.5.0-...) 11.5.0
```

### Step 2: Isaac Simのソースビルド

```bash
# 作業ディレクトリの作成
mkdir -p ~/isaac && cd ~/isaac

# リポジトリのクローン
git clone --depth=1 --recursive https://github.com/isaac-sim/IsaacSim
cd IsaacSim

# Git LFSファイルの取得
git lfs install && git lfs pull

# ビルド実行
./build.sh
```

ビルドには数分〜十数分かかります。`BUILD (RELEASE) SUCCEEDED` と表示されれば成功です。

### Step 3: 環境変数の設定

```bash
export ISAACSIM_PATH="${PWD}/_build/linux-aarch64/release"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
```

### Step 4: Isaac Labのインストール

```bash
cd ~/isaac

# リポジトリのクローン
git clone --recursive https://github.com/isaac-sim/IsaacLab
cd IsaacLab

# Isaac Simへのシンボリックリンク作成
ln -sfn "${ISAACSIM_PATH}" "${PWD}/_isaac_sim"

# インストール
./isaaclab.sh --install
```

### Step 5: 動作確認

```bash
# 必要な環境変数
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"

# Cartpole（棒立てバランス）タスクで学習テスト
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Cartpole-v0 \
  --headless \
  --max_iterations 100
```

---

## 5. 学習結果

### Cartpole（棒立てバランス）タスク

| 指標 | 値 |
|-----|-----|
| 学習時間 | 21.2秒（100イテレーション） |
| 計算速度 | 290,312 steps/s |
| 最終報酬 | 4.94 |
| エピソード長 | 300（最大値=成功） |

```
Learning iteration 99/100
Computation: 290312 steps/s (collection: 0.152s, learning 0.073s)
Mean reward: 4.94
Mean episode length: 300.00
Episode_Termination/time_out: 1.0000
Episode_Termination/cart_out_of_bounds: 0.0000
```

エピソード長が最大値（300）に達し、`cart_out_of_bounds`が0であることから、棒を倒さずにバランスを取る学習が成功しています。

---

## 6. ディレクトリ構成

```
~/isaac/
├── IsaacSim/
│   ├── _build/
│   │   └── linux-aarch64/
│   │       └── release/      # ビルド済みIsaac Sim
│   └── source/
└── IsaacLab/
    ├── _isaac_sim -> ../IsaacSim/_build/linux-aarch64/release
    ├── scripts/
    │   └── reinforcement_learning/
    │       └── rsl_rl/
    │           └── train.py  # 学習スクリプト
    └── logs/                  # 学習ログ出力先
```

---

## 7. 利用可能なタスク

Isaac Labには多数の学習タスクが用意されています。

| タスク | 説明 |
|-------|------|
| `Isaac-Cartpole-v0` | 棒立てバランス（入門用） |
| `Isaac-Ant-v0` | 四足歩行ロボット |
| `Isaac-Humanoid-v0` | 二足歩行ロボット |
| `Isaac-Velocity-Rough-H1-v0` | H1ロボットの歩行学習 |

タスク一覧は以下で確認できます:

```bash
./isaaclab.sh -p scripts/tools/list_envs.py
```

---

## 8. 学習結果の確認

学習ログはTensorBoardで可視化できます:

```bash
tensorboard --logdir logs/
```

ブラウザで `http://localhost:6006` にアクセスすると、報酬の推移やロスの変化を確認できます。

---

## 9. 注意事項

### GUI表示について

DGX Spark（aarch64）では、現時点でGUI表示に制限があります。

| モード | 状態 |
|-------|------|
| ヘッドレス（`--headless`） | 正常動作 |
| GUI表示（`--headless`なし） | 強制終了される |
| ライブストリーミング | aarch64非対応 |

実際にGUIモードで実行を試みたところ、以下のエラーで強制終了されました:

```
[Warning] [omni.fabric.plugin] Warning: attribute viewportHandle not found for bucket id 9
/home/.../python.sh: 73 行: 強制終了
There was an error running python
```

現状の運用方法:
- **学習**: ヘッドレスモードで実行
- **結果確認**: TensorBoardでログを可視化
- **動画出力**: `--video`オプションで録画（要検証）

最新情報は[公式ドキュメント](https://build.nvidia.com/spark/isaac/isaac-lab)を参照してください。

### 環境変数の永続化

毎回設定が必要な環境変数は、`.bashrc`に追記すると便利です:

```bash
echo 'export ISAACSIM_PATH="$HOME/isaac/IsaacSim/_build/linux-aarch64/release"' >> ~/.bashrc
echo 'export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"' >> ~/.bashrc
echo 'export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"' >> ~/.bashrc
```

---

## まとめ

DGX Spark（Blackwell GB10）上でIsaac Labを動作させ、強化学習タスクの実行に成功しました。

| 項目 | 結果 |
|-----|------|
| Isaac Simソースビルド | 成功（約8.5分） |
| Isaac Labインストール | 成功 |
| Cartpole学習 | 成功（290K steps/s） |
| ヘッドレス実行 | 問題なし |
| GUI表示 | 現時点で非対応 |

128GB UMAと高性能なTensorコアにより、大規模な並列シミュレーションが1台のデスクサイド機で実現可能です。GUI表示の制限はありますが、ヘッドレスでの学習パイプラインは問題なく動作します。

---

## 参考リンク

- [Install and Use Isaac Sim and Isaac Lab | DGX Spark](https://build.nvidia.com/spark/isaac/overview)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/)
- [Isaac Lab GitHub](https://github.com/isaac-sim/IsaacLab)
