#pragma once
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <iostream>
#include <cassert>
#include <codecvt> 
#include "ctype.h"
#include "sampler.h"

#define PI 3.14159265358979323846	// 直書き

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
		bool _initilized;
		int _coverage;
		int _c_max;
		int _t_max;
		int _num_characters;
		GLM(){ }
		// t以前の何文字から素性ベクトルを作るか
		// c_maxはc_iのiの範囲（0 ≤ i ≤ c_max）
		// t_maxはt_iのiの範囲（0 ≤ i ≤ t_max）
		// 論文ではcoverage = 8, c_max = 1, t_max = 4
		GLM(int coverage, int c_max, int t_max){
			_coverage = coverage;
			_c_max = c_max;
			_t_max = t_max;
			_num_characters = 0;
			_initilized = false;
		}
		~GLM(){
			if(_initilized == false){
				return;
			}
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
			_wr_cont = new double[_coverage];
			_wr_ch = new double[_coverage];
			for(int i = 0;i < _coverage;i++){
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
			_wp_cont = new double[_coverage];
			_wp_ch = new double[_coverage];
			for(int i = 0;i < _coverage;i++){
				_wp_cont[i] = sampler::normal(0, 1);
				_wp_ch[i] = sampler::normal(0, 1);
			}
			_num_characters = num_characters;
			_initilized = true;
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
		double compute_log_weight_prior(double w){
			return -0.5 * w * w - log(sqrt(2.0 * PI));
		}
		double compute_r(const int* feature){
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
		double compute_p(const int* feature){
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
		double compute_cumulative_probability(double l, double r, double p){
			return 1 - boost::math::ibeta(l + 1, r, p);
		}
		template <class Archive>
		void serialize(Archive &ar, unsigned int version){
			boost::serialization::split_free(ar, *this, version);
		}
		void save(std::string filename){
			std::ofstream ofs(filename);
			boost::archive::binary_oarchive oarchive(ofs);
			oarchive << *this;
		}
		bool load(std::string filename){
			std::ifstream ifs(filename);
			if(ifs.good()){
				boost::archive::binary_iarchive iarchive(ifs);
				iarchive >> *this;
				return true;
			}
			return false;
		}
	};
} // namespace npycrf

// モデルの保存用
namespace boost { 
	namespace serialization {
		template<class Archive>
		void save(Archive &ar, const npycrf::GLM &glm, unsigned int version) {
			ar & glm._initilized;
			ar & glm._num_characters;
			ar & glm._coverage;
			ar & glm._c_max;
			ar & glm._t_max;
			if(glm._initilized){
				int c_max = glm._c_max;
				int t_max = glm._t_max;
				int coverage = glm._coverage;
				int num_characters = glm._num_characters;
				int num_types = CTYPE_TOTAL_TYPE;
				// r
				ar & glm._wr_bias;
				for(int i = 0;i <= c_max;i++){ for(int j = 0;j < num_characters + 1;j++){
					ar & glm._wr_c[i][j];
				}}
				for(int i = 0;i <= t_max;i++){	for(int j = 0;j < num_types;j++){
					ar & glm._wr_t[i][j];
				}}
				for(int i = 0;i < coverage;i++){
					ar & glm._wr_cont[i];
					ar & glm._wr_ch[i];
				}
				// p
				ar & glm._wp_bias;
				for(int i = 0;i <= c_max;i++){ for(int j = 0;j < num_characters + 1;j++){
					ar & glm._wp_c[i][j];
				}}
				for(int i = 0;i <= t_max;i++){	for(int j = 0;j < num_types;j++){
					ar & glm._wp_t[i][j];
				}}
				for(int i = 0;i < coverage;i++){
					ar & glm._wp_cont[i];
					ar & glm._wp_ch[i];
				}
			}
		}
		template<class Archive>
		void load(Archive &ar, npycrf::GLM &glm, unsigned int version) {
			ar & glm._initilized;
			ar & glm._num_characters;
			ar & glm._coverage;
			ar & glm._c_max;
			ar & glm._t_max;
			if(glm._initilized){
				int c_max = glm._c_max;
				int t_max = glm._t_max;
				int coverage = glm._coverage;
				int num_characters = glm._num_characters;
				int num_types = CTYPE_TOTAL_TYPE;
				// r
				glm._wr_c = new double*[c_max + 1];
				for(int i = 0;i <= c_max;i++){
					glm._wr_c[i] = new double[num_characters + 1];
				}
				glm._wr_t = new double*[t_max + 1];
				for(int i = 0;i <= t_max;i++){
					glm._wr_t[i] = new double[num_types];
				}
				glm._wr_cont = new double[coverage];
				glm._wr_ch = new double[coverage];
				// p
				glm._wp_c = new double*[c_max + 1];
				for(int i = 0;i <= c_max;i++){
					glm._wp_c[i] = new double[num_characters + 1];
				}
				glm._wp_t = new double*[t_max + 1];
				for(int i = 0;i <= t_max;i++){
					glm._wp_t[i] = new double[num_types];
				}
				glm._wp_cont = new double[coverage];
				glm._wp_ch = new double[coverage];
				// r
				ar & glm._wr_bias;
				for(int i = 0;i <= c_max;i++){ for(int j = 0;j < num_characters + 1;j++){
					ar & glm._wr_c[i][j];
				}}
				for(int i = 0;i <= t_max;i++){	for(int j = 0;j < num_types;j++){
					ar & glm._wr_t[i][j];
				}}
				for(int i = 0;i < coverage;i++){
					ar & glm._wr_cont[i];
					ar & glm._wr_ch[i];
				}
				// p
				ar & glm._wp_bias;
				for(int i = 0;i <= c_max;i++){ for(int j = 0;j < num_characters + 1;j++){
					ar & glm._wp_c[i][j];
				}}
				for(int i = 0;i <= t_max;i++){	for(int j = 0;j < num_types;j++){
					ar & glm._wp_t[i][j];
				}}
				for(int i = 0;i < coverage;i++){
					ar & glm._wp_cont[i];
					ar & glm._wp_ch[i];
				}
			}
		}
	}
} // namespace boost::serialization