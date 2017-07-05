#pragma once
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <iostream>
#include <cassert>
#include "ctype.h"
using namespace std;
#include <codecvt> 

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
			int length = get_feature_vector_length();
			int* features = new int[length];
			for(int i = 0;i < length;i++){
				features[i] = 0;
			}
			wchar_t character;
			int char_id;
			// 1文字目
			character = word[word.size() - 1];
			char_id = get_character_id(character);
			if(char_id > 0){    // 訓練データにないものは無視
				features[0] = char_id;
			}
			// 2文字目
			if(word.size() > 1){
				character = word[word.size() - 2];
				char_id = get_character_id(character);
				if(char_id > 0){   // 訓練データにないものは無視
					features[1] = char_id;
				}
			}
			// 1文字目
			character = word[word.size() - 1];
			return features;
		}
		int get_character_id(wchar_t character){
			auto itr = _char_ids.find(character);
			if(itr == _char_ids.end()){
				return -1;
			}
			return itr->second;
		}
		int get_feature_vector_length(){
			int length = 0;
			// character at time t−i (0 ≤ i ≤ 1)
			length += 2;
			// character type at time t−i (0 ≤ i ≤ 4)
			length += 5;
			// # of the same character types before t
			length += 1;
			// # of times character types changed
			length += 1;
			return length;
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
		void dump_characters(){
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