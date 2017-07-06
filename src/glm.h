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

using namespace std;

// Inducing Word and Part-of-Speech with Pitman-Yor Hidden Semi-Markov Models
// http://chasen.org/~daiti-m/paper/acl2015pyhsmm.pdf

namespace npycrf{
	class GLM{
	public:
		int _coverage;
		int _c_max;
		int _t_max;
		unordered_set<wstring> _word_set;
		unordered_map<wchar_t, int> _char_ids;
		vector<std::pair<int, int*>> _length_features_pair;
		double _w_bias;      // バイアス
		double** _w_c;
		double** _w_t;
		double* _w_cont;
		double* _w_ch;
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
		void init_weights(){
			int num_characters = _char_ids.size();
			int num_types = 280;	// Unicode
			_w_bias = sampler::normal(0, 1);
			_w_c = new double*[_c_max + 1];
			for(int i = 0;i <= _c_max;i++){
				_w_c[i] = new double[num_characters];
				for(int j = 0;j < num_characters;j++){
					_w_c[i][j] = sampler::normal(0, 1);
				}
			}
			_w_t = new double*[_t_max + 1];
			for(int i = 0;i <= _c_max;i++){
				_w_t[i] = new double[num_types];
				for(int j = 0;j < num_types;j++){
					_w_t[i][j] = sampler::normal(0, 1);
				}
			}
			_w_cont = new double[_coverage - 1];
			_w_ch = new double[_coverage - 1];
			for(int i = 0;i < _coverage - 1;i++){
				_w_cont[i] = sampler::normal(0, 1);
				_w_ch[i] = sampler::normal(0, 1);
			}
		}
		void compile(){
			std::pair<int, int*> pair;
			for(auto word: _word_set){
				int word_length = word.size();
				int* features = extract_features(word);
				pair.first = word_length;
				pair.second = features;
				_length_features_pair.push_back(pair);
			}
			init_weights();
		}
		int* extract_features(wstring &word){
			int num_features = get_num_features();
			int* features = new int[num_features];
			for(int i = 0;i < num_features;i++){
				features[i] = 0;
			}
			wchar_t character;
			int char_id;
			int t = word.size() - 1;
			// 文字による素性
			for(int i = 0;i <= _c_max;i++){
				if(t - i < 0){
					break;
				}
				character = word[t - i];
				char_id = get_character_id(character);
				if(char_id > 0){    // 訓練データにないものは無視
					features[i] = char_id;
				}
			}
			// 文字種による素性
			// Unicodeでは全280種
			for(int i = 0;i <= _t_max;i++){
				if(t - i < 0){
					break;
				}
				character = word[t - i];
				unsigned int type = chartype::get_type(character);
				features[i + _c_max + 1] = type;
			}
			// t以前の同じ文字種の数
			int cont = 0;
			unsigned int basetype = chartype::get_type(word[t]);
			for(int i = 1;i < _coverage;i++){
				if(t - i < 0){
					break;
				}
				character = word[t - i];
				unsigned int type = chartype::get_type(character);
				if(type == basetype){
					cont += 1;
				}
			}
			features[_c_max + _t_max + 2] = cont;
			// 文字種が変わった数
			int ch = 0;
			for(int i = 1;i < _coverage;i++){
				if(t - i < 0){
					break;
				}
				character = word[t - i];
				unsigned int type = chartype::get_type(character);
				if(type != basetype){
					ch += 1;
					basetype = type;
				}
			}
			features[_c_max + _t_max + 3] = ch;
			return features;
		}
		int get_character_id(wchar_t character){
			auto itr = _char_ids.find(character);
			if(itr == _char_ids.end()){
				return -1;
			}
			return itr->second;
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
		void add_words(vector<wstring> &words){
			for(auto word: words){
				auto itr = _word_set.find(word);
				if(itr == _word_set.end()){
					_word_set.insert(word);
					add_character(word);
				}
			}
		}
		void add_character(wstring &word){
			for(auto character: word){
				auto itr = _char_ids.find(character);
				if(itr == _char_ids.end()){
					_char_ids[character] = _char_ids.size() + 1;    // 0避け
				}
			}
		}
		void dump_words(){
			cout << "word	feature" << endl;
			int num_features = get_num_features();
			for(auto word: _word_set){
				int* feature = extract_features(word);
				wcout << word << "	";
				for(int i = 0;i < num_features;i++){
					wcout << feature[i] << ", ";
				}
				wcout << endl;
				delete[] feature;
			}
		}
		void dump_characters(){
			cout << "char	id	type" << endl;
			for(auto elem: _char_ids){
				wchar_t character = elem.first;
				unsigned int type = chartype::get_type(character);
				std::wstring_convert<std::codecvt_utf8<wchar_t>,wchar_t> cv;
				std::wstring name = cv.from_bytes(chartype::get_name(type));
				wcout << character << "	" << elem.second << "	" << name << endl;
			}
		}
	};
} // namespace npycrf