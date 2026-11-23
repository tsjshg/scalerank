# ScaleRank

[English](README.md)

近傍の**順位**を、いろいろな大きさのスケールで保つ次元削減です。点ごとに近傍の大きさ K を K^−γ に比例する確率で選び、その近傍の内側の点 1 つと外側の点いくつかを取って、内側の点のほうが近くなるように地図を動かします。γ = 0 はすべての大きさを同じ重みで扱うので全体の配置を、γ が大きいと小さい近傍に集中するので局所の構造を重視します。正確な順位は各点の近傍だけで計算し、それより大きい順位はランダムな部分集合から推定するので、N × N の表は作りません。

densMAP にならった密度項を足すこともできます。各点の局所的な半径をデータのものに合わせる項で、順位だけの損失では局所的な大きさが伝わらないのを補います。

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
