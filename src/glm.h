#pragma once
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <iostream>
#include <cassert>
using namespace std;

// Inducing Word and Part-of-Speech with Pitman-Yor Hidden Semi-Markov Models
// http://chasen.org/~daiti-m/paper/acl2015pyhsmm.pdf

namespace npycrf{
    class GLM{
    private:
        int _coverage;
        unordered_set<wstring> _word_set;
        unordered_map<wchar_t, int> _char_ids;
        vector<std::pair<int, unsigned int*>> _length_feature_pair;
    public:
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
                int* features = generate_feature(word);
            }
        }
        int* generate_feature(wstring &word){
            int length = get_feature_vector_length();
            unsigned int* features = new unsigned int[length];
            for(int i = 0;i < length;i++){
                features[i] = 0;
            }
            features[0] = 1; // バイアス
            int pos = 1;
            wchar_t character;
            int char_id;
            // 1文字目
            character = word[word.size() - 1];
            char_id = get_character_id(character);
            assert(char_id != -1);
            vec[pos + char_id] = 1;
            pos += char_id;
            // 2文字目
            if(word.size() > 1){
                character = word[word.size() - 2];
                char_id = get_character_id(character);
                assert(char_id != -1);
                vec[pos + char_id] = 1;
                pos += char_id;
            }
            // 1文字目
            character = word[word.size() - 1];
            return vec;
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
            // f0
            length += 1;
            // character at time t−i (0 ≤ i ≤ 1)
            length += 2 * _char_ids.size();
            // character type at time t−i (0 ≤ i ≤ 4)
            length += 5;
            // # of the same character types before t
            length += _coverage;
            // # of times character types changed
            length += _coverage;
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
                    _char_ids[character] = _char_ids.size();
                }
            }
        }
        void dump_characters(){
            for(auto elem: _char_ids){
                wcout << elem.first << ": " << elem.second << endl;
            }
        }
    };
} // namespace npycrf