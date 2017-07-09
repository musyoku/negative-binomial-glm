## 負の二項分布の一般化線形モデルによる最大単語長の予測

Negative Binomial Generalized Linear Modelによる最大単語長の予測の実験コードです。

半教師あり形態素解析でラティスの圧縮に用います。

文字列のある文字に注目すると、この文字はある単語の一部になっていますが、その単語の最大長を予測し可能な単語分割を絞ることで単語分割のサンプリングを高速化することができるそうです。

## 使用

### 動作に必要なもの

- C++ コンパイラ
	- GCC と Apple LLVM で動作を確認しています

- Boost
	- 不完全ベータ関数で使います

### ビルドと実行

```
make train
make test
```

```
./train
./test
```

### パラメータ

- coverage
	- tから何文字を対象に素性ベクトルを作るか
	- contとchに影響する

- c_max
	- 素性にする文字をtから何文字にするか

- t_max
	- tから何文字までの文字種を素性にするか

- sigma
	- MCMCのランダムウォーク時のノイズの標準偏差

重みの事前分布は平均0、分散1の正規分布で固定です。

## 結果

ネットで収集したテキスト78万行のうち10万行を学習、68万行をテストに用いました。

以下は全てテストデータでの結果です。

学習時間がまだ足りていないので精度はさらに上がると思います。

### 真の単語長ごとの予測精度

```
L	Precision 
1:	1
2:	0.999942
3:	0.999797
4:	0.998967
5:	0.963054
6:	0.946659
7:	0.9385
8:	0.887836
9:	0.847732
10:	0.777372
11:	0.731287
12:	0.678965
13:	0.518164
14:	0.386861
15:	0.227898
16:	0.194595
n ≥ 5:	0.932351
all:	0.9973
```

L: 真の単語長

### 予測単語長の頻度

```
L	Frequency
1:	675
2:	3836
3:	236045
4:	1689464
5:	1460799
6:	1573116
7:	870510
8:	302536
9:	194378
10:	113467
11:	62724
12:	29612
13:	23365
14:	13641
15:	7244
16:	18878
```

L: 予測単語長

### 真の単語長の分布

```
L	Frequency
1:	4027
2:	30147
3:	32146
4:	30116
5:	17915
6:	12450
7:	8830
8:	6471
9:	4008
10:	2691
11:	1712
12:	1142
13:	739
14:	487
15:	367
16:	297
```

L: 真の単語長

### 予測と真の単語長との誤差

```
L	Mean		StdDev
1:	3.47967:	0.98517
2:	4.06068:	1.01883
3:	3.95587:	1.63497
4:	3.85345:	2.08927
5:	3.70594:	2.51064
6:	3.79879:	2.50787
7:	3.72924:	2.44623
8:	5.00212:	3.28827
9:	3.55888:	3.36619
10:	2.26669:	3.58750
11:	1.11176:	3.49680
12:	0.26156:	3.56151
13:	-1.43499:	3.73668
14:	-2.79416:	3.94783
15:	-4.09627:	3.89203
16:	-5.05135:	4.11721
```

L: 真の単語長

## 参考文献

- [Inducing Word and Part-of-Speech with Pitman-Yor Hidden Semi-Markov Models](http://chasen.org/~daiti-m/paper/acl2015pyhsmm.pdf)