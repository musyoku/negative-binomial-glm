CC = g++
INCLUDE = -I`python -c 'from distutils.sysconfig import *; print get_python_inc()'`
BOOST = -lboost_python -lpython2.7 -lboost_serialization
CFLAGS_SO = -shared -fPIC -std=c++11 -L/usr/local/lib -O3 

python: ## Python用にビルド
	$(CC) ../model.cpp -o model.so $(INCLUDE) $(CFLAGS_SO) $(BOOST)