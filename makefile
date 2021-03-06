CC = g++
BOOST = -lboost_serialization
CFLAGS = -std=c++11 -L/usr/local/lib -O3

.PHONY: train
train: ## 学習用
	$(CC) train.cpp -o train $(CFLAGS) $(BOOST)

.PHONY: test
test: ## テスト用
	$(CC) test.cpp -o test $(CFLAGS) $(BOOST)

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	
.DEFAULT_GOAL := help