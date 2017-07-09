## 負の二項分布の一般化線形モデルによる単語長の予測

Negative Binomial Generalized Linear Modelによる単語長の予測の実験コードです。

半教師あり形態素解析でラティスの圧縮に用います。

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

### 真の単語長ごとの予測精度

```
L	Precision 
1:	1
2:	0.999884
3:	0.99954
4:	0.996731
5:	0.959291
6:	0.939219
7:	0.903648
8:	0.864059
9:	0.808398
10:	0.750241
11:	0.719903
12:	0.695485
13:	0.523901
14:	0.443796
15:	0.324165
16:	0.278378
n ≥ 5:	0.921061
total:	0.996705
```

### 予測単語長の頻度

```
L	Frequency
1:	1252
2:	10282
3:	278366
4:	1556399
5:	1515476
6:	1538061
7:	851951
8:	327790
9:	191175
10:	122895
11:	73463
12:	41100
13:	23074
14:	20110
15:	11141
16:	37755
```

### 予測と真の単語長との誤差

```
L	Mean		StdDev
1:	3.54995:	1.11823
2:	4.13058:	1.21012
3:	4.07473:	1.97458
4:	3.91065:	2.35720
5:	4.02730:	2.89934
6:	3.60888:	2.76102
7:	3.90228:	2.96128
8:	4.98507:	3.50432
9:	3.76038:	3.69712
10:	2.55571:	3.93372
11:	1.28734:	3.78263
12:	0.54240:	3.84449
13:	-1.37859:	4.11708
14:	-2.56788:	4.32276
15:	-3.86248:	4.18538
16:	-4.90811:	4.32431
```

Lは真の単語長

## 参考文献

- [Inducing Word and Part-of-Speech with Pitman-Yor Hidden Semi-Markov Models](http://chasen.org/~daiti-m/paper/acl2015pyhsmm.pdf)