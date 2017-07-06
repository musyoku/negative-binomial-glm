#pragma once
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <iostream>
#include <cassert>
#include <codecvt> 
#include "ctype.h"

using namespace std;

// Inducing Word and Part-of-Speech with Pitman-Yor Hidden Semi-Markov Models
// http://chasen.org/~daiti-m/paper/acl2015pyhsmm.pdf

namespace npycrf{
	class GLM{
	public:
		int _coverage;
		unordered_set<wstring> _word_set;
		unordered_map<wchar_t, int> _char_ids;
		vector<std::pair<int, int*>> _length_features_pair;
		double w_bias;      // バイアス
		double* w_c;
		double* w_t;
		double* w_cont;
		double* w_ch;
		GLM(){
			_coverage = 0;
		}
		// 論文ではcoverage = 8
		// t以前の何文字から素性ベクトルを作るか
		GLM(int coverage){
			_coverage = coverage;
		}
		void compile(){
			std::pair<int, int*> pair;
			for(auto word: _word_set){
				int word_length = word.size();
				int* features = extract_features(word);
			}
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
			for(int i = 0;i < 2;i++){
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
			for(int i = 0;i < 5;i++){
				if(t - i < 0){
					break;
				}
				character = word[t - i];
				unsigned int type = chartype::get_type(character);
				features[i + 2] = type;
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
			features[7] = cont;
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
			features[8] = ch;
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
			// character at time t−i (0 ≤ i ≤ 1)
			num += 2;
			// character type at time t−i (0 ≤ i ≤ 4)
			num += 5;
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