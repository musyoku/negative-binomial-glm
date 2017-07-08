#pragma once
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <iostream>
#include <cassert>
#include <codecvt> 
#include "ctype.h"
#include "sampler.h"

// Inducing Word and Part-of-Speech with Pitman-Yor Hidden Semi-Markov Models
// http://chasen.org/~daiti-m/paper/acl2015pyhsmm.pdf

namespace npycrf{
	double sigmoid(double x){
		return 1 / (1 + exp(-x));
	}
	unsigned int factorial(unsigned int n) {
	    if (n == 0){
	       return 1;
	    }
	    return n * factorial(n - 1);
	}
	class GLM{
	public:
		double _wr_bias;      // バイアス
		double** _wr_c;
		double** _wr_t;
		double* _wr_cont;
		double* _wr_ch;
		double _wp_bias;      // バイアス
		double** _wp_c;
		double** _wp_t;
		double* _wp_cont;
		double* _wp_ch;
		int _coverage;
		int _c_max;
		int _t_max;
		GLM(){ }
		// t以前の何文字から素性ベクトルを作るか
		// c_maxはc_iのiの範囲（0 ≤ i ≤ c_max）
		// t_maxはt_iのiの範囲（0 ≤ i ≤ t_max）
		// 論文ではcoverage = 8, c_max = 1, t_max = 4
		GLM(int coverage, int c_max, int t_max){
			_coverage = coverage;
			_c_max = c_max;
			_t_max = t_max;
		}
		~GLM(){
			for(int i = 0;i <= _c_max;i++){
				delete[] _wr_c[i];
				delete[] _wp_c[i];
			}
			for(int i = 0;i <= _t_max;i++){
				delete[] _wr_t[i];
				delete[] _wp_t[i];
			}
			delete[] _wr_c;
			delete[] _wp_c;
			delete[] _wr_t;
			delete[] _wp_t;
			delete[] _wr_cont;
			delete[] _wr_ch;
			delete[] _wp_cont;
			delete[] _wp_ch;
		}
		void init_weights(int num_characters){
			int num_types = CTYPE_TOTAL_TYPE;	// Unicode
			// r
			_wr_bias = sampler::normal(0, 1);
			_wr_c = new double*[_c_max + 1];
			for(int i = 0;i <= _c_max;i++){
				_wr_c[i] = new double[num_characters + 1];
				_wr_c[i][0] = 0;
				for(int j = 1;j < num_characters + 1;j++){
					_wr_c[i][j] = sampler::normal(0, 1);
				}
			}
			_wr_t = new double*[_t_max + 1];
			for(int i = 0;i <= _t_max;i++){
				_wr_t[i] = new double[num_types];
				_wr_t[i][0] = 0;
				for(int j = 1;j < num_types;j++){
					_wr_t[i][j] = sampler::normal(0, 1);
				}
			}
			_wr_cont = new double[_coverage - 1];
			_wr_ch = new double[_coverage - 1];
			for(int i = 0;i < _coverage - 1;i++){
				_wr_cont[i] = sampler::normal(0, 1);
				_wr_ch[i] = sampler::normal(0, 1);
			}
			// p
			_wp_bias = sampler::normal(0, 1);
			_wp_c = new double*[_c_max + 1];
			for(int i = 0;i <= _c_max;i++){
				_wp_c[i] = new double[num_characters + 1];
				_wp_c[i][0] = 0;
				for(int j = 1;j < num_characters + 1;j++){
					_wp_c[i][j] = sampler::normal(0, 1);
				}
			}
			_wp_t = new double*[_t_max + 1];
			for(int i = 0;i <= _t_max;i++){
				_wp_t[i] = new double[num_types];
				_wp_t[i][0] = 0;
				for(int j = 1;j < num_types;j++){
					_wp_t[i][j] = sampler::normal(0, 1);
				}
			}
			_wp_cont = new double[_coverage - 1];
			_wp_ch = new double[_coverage - 1];
			for(int i = 0;i < _coverage - 1;i++){
				_wp_cont[i] = sampler::normal(0, 1);
				_wp_ch[i] = sampler::normal(0, 1);
			}
		}
		int get_num_features(){
			int num = 0;
			// character at time t−i (0 ≤ i ≤ c_max)
			num += _c_max + 1;
			// character type at time t−i (0 ≤ i ≤ t_max)
			num += _t_max + 1;
			// # of the same character types before t
			num += 1;
			// # of times character types changed
			num += 1;
			return num;
		}
		double compute_nb_log_likelihood(double l, double r, double p){
			double likelihood = 0;
			likelihood += lgamma(r + l);
			likelihood -= lgamma(r) + lgamma(l + 1);
			likelihood += l * log(p);
			likelihood += r * log(1 - p);
			return likelihood;
		}
		double compute_r(int* feature){
			double u = _wr_bias;
			for(int i = 0;i <= _c_max;i++){
				u += _wr_c[i][feature[i]];
			}
			for(int i = 0;i <= _t_max;i++){
				u += _wr_t[i][feature[i + _c_max + 1]];
			}
			u += _wr_cont[feature[_c_max + _t_max + 2]];	// cont
			u += _wr_cont[feature[_c_max + _t_max + 3]];	// ch
			return exp(u);
		}
		double compute_p(int* feature){
			double u = _wp_bias;
			for(int i = 0;i <= _c_max;i++){
				u += _wp_c[i][feature[i]];
			}
			for(int i = 0;i <= _t_max;i++){
				u += _wp_t[i][feature[i + _c_max + 1]];
			}
			u += _wp_cont[feature[_c_max + _t_max + 2]];	// cont
			u += _wp_cont[feature[_c_max + _t_max + 3]];	// ch
			return sigmoid(u);
		}
	};
} // namespace npycrf