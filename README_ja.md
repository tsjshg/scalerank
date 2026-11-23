# ScaleRank

[English](README.md)

**クラスタの中も、クラスタどうしの関係も読める地図。**

t-SNE や UMAP は各点のすぐ近くの近傍を保ちます。ただし、その地図でクラスタ間の距離や並び方を読んではいけない、と指摘されています（[Chari & Pachter 2023](https://doi.org/10.1371/journal.pcbi.1011288)）。PCA は並び方を保ちますが、近傍はぼやけます。ScaleRank はその両方を保ちます。

- **両方で負けない.** [drbench](https://github.com/tsjshg/drbench) の 8 データすべてで、局所と大域の両方で ScaleRank を上回る手法はありませんでした。パラメータを 30 通り調整した t-SNE・UMAP も同じです。
- **全体の配置が残る.** データと地図の距離相関は 0.79 です（t-SNE 0.52、UMAP 0.37）。近傍の保存は t-SNE に近い値です（0.51 と 0.54）。
- **配置は seed ではなくデータで決まる.** 乱数の seed を変えても、全体の配置の一致は 0.97 です（UMAP 0.78）。
- **密度も保てる.** 密度項を足すと、点の密集度も保ちます。Open Problems の次元削減タスクでは 0.491 で、同じ条件で回した比較手法（densMAP 0.452、t-SNE 0.412、UMAP 0.373）をすべて上回りました（[詳細](research/results/openproblems/README.md)）。

局所だけなら t-SNE がまだ少し上で、7 万点ではその差が広がります。

### しくみ

近傍の**順位**を、いろいろな大きさのスケールで保ちます。

1. 点ごとに、近傍の大きさ K を K^−γ に比例する確率で選ぶ。
2. その近傍の内側の点 1 つと、外側の点いくつかを取る。
3. 内側の点のほうが近くなるように地図を動かす。

γ = 0 はすべての大きさを同じ重みで扱うので全体の配置を、γ が大きいと小さい近傍に集中するので局所の構造を重視します。正確な順位は各点の近傍だけで計算し、それより大きい順位はランダムな部分集合から推定するので、N × N の表は作りません。

順位だけの損失では、局所的な大きさが伝わりません。densMAP にならった密度項で、各点の局所的な半径をデータのものに合わせます。

- ブラウザのアプリ: <https://tsjshg.github.io/scalerank-web/>（[ソース](https://github.com/tsjshg/scalerank-web)）
- 研究ノート: <https://tsjshg.github.io/scalerank-web/docs/ja.html>
- ベンチマーク: [drbench](https://github.com/tsjshg/drbench)。Open Problems での結果: [research/results/openproblems](research/results/openproblems/README.md)

## Python

```bash
pip install scalerank
```

```python
from scalerank import ScaleRank

Y = ScaleRank(gamma=0.5, random_state=0).fit_transform(X)   # X: (n, d) の float 配列
```

主なパラメータと構成は [README.md](README.md) を見てください。
