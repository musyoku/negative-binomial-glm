負の二項分布の一般化線形モデルによる単語長の予測の実験コードです。

## 動作に必要なもの

- C++ コンパイラ
	- GCC と Apple LLVM で動作を確認しています

- Boost
	- 不完全ベータ関数で使います

## インストールと実行

```
make train
make test
```

```
./train
./test
```

## 参考文献

- [Inducing Word and Part-of-Speech with Pitman-Yor Hidden Semi-Markov Models](http://chasen.org/~daiti-m/paper/acl2015pyhsmm.pdf)