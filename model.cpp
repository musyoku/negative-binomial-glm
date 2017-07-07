#include <boost/python.hpp>
#include <fstream>
#include <vector>
#include <cassert>
#include "src/glm.h"
using namespace std;
using namespace boost;
using namespace npycrf;

void split_word_by(const wstring &str, wchar_t delim, vector<wstring> &words){
	words.clear();
	wstring item;
	for(wchar_t ch: str){
		if (ch == delim){
			if (!item.empty()){
				words.push_back(item);
			}
			item.clear();
		}else{
			item += ch;
		}
	}
	if (!item.empty()){
		words.push_back(item);
	}
}

class Indices{
private:
	vector<int> _indices;
public:
	int size(){
		return _indices.size();
	}
    int operator[](int i) {
    	assert(i < size());
    	return _indices[i];
    }
    void add(int i){
    	_indices.push_back(i);
    }
};

class PyTrainer{
private:
	vector<std::pair<int, int*>> _length_features_pair;
	bool _compiled;
	// 重みの変更の影響を受ける素性ベクトルをリストアップ
	Indices*** _indices_wr_c;
	Indices*** _indices_wr_t;
	Indices** _indices_wr_cont;
	Indices** _indices_wr_ch;
	Indices*** _indices_wp_c;
	Indices*** _indices_wp_t;
	Indices** _indices_wp_cont;
	Indices** _indices_wp_ch;
public:
	GLM* _glm;
	unordered_set<wstring> _word_set;
	unordered_map<wchar_t, int> _char_ids;
	int _coverage;
	int _c_max;
	int _t_max;

	PyTrainer(int coverage, int c_max, int t_max){
		_glm = new GLM(coverage, c_max, t_max);
		_coverage = coverage;
		_c_max = c_max;
		_t_max = t_max;
		_compiled = false;
		_indices_wr_c = NULL;
		_indices_wr_t = NULL;
		_indices_wr_cont = NULL;
		_indices_wr_ch = NULL;
		_indices_wp_c = NULL;
		_indices_wp_t = NULL;
		_indices_wp_cont = NULL;
		_indices_wp_ch = NULL;
	}
	~PyTrainer(){
		for(auto pair: _length_features_pair){
			delete[] pair.second;
		}
		delete _glm;
		if(_compiled){
			int num_characters = _char_ids.size();
			int num_types = 280;	// Unicode
			for(int i = 0;i <= _c_max;i++){
				for(int j = 0;j < num_characters;j++){
					delete _indices_wr_c[i][j];
					delete _indices_wp_c[i][j];
				}
				delete[] _indices_wr_c[i];
				delete[] _indices_wp_c[i];
			}
			delete[] _indices_wr_c;
			delete[] _indices_wp_c;
			for(int i = 0;i <= _t_max;i++){
				for(int j = 0;j < num_types;j++){
					delete _indices_wr_t[i][j];
					delete _indices_wp_t[i][j];
				}
				delete[] _indices_wr_t[i];
				delete[] _indices_wp_t[i];
			}
			delete[] _indices_wr_t;
			delete[] _indices_wp_t;
			for(int i = 0;i < _coverage - 1;i++){
				delete _indices_wr_cont[i];
				delete _indices_wp_cont[i];
				delete _indices_wr_ch[i];
				delete _indices_wp_ch[i];
			}
			delete[] _indices_wr_cont;
			delete[] _indices_wp_cont;
			delete[] _indices_wr_ch;
			delete[] _indices_wp_ch;
		}
	}
	void add_textfile(string filename){
		wifstream ifs(filename.c_str());
		wstring sentence;
		assert(ifs.fail() == false);
		while (getline(ifs, sentence)){
			if(sentence.empty()){
				continue;
			}
			if (PyErr_CheckSignals() != 0) {		// ctrl+cが押されたかチェック
				return;
			}
			vector<wstring> words;
			split_word_by(sentence, L' ', words);
			add_words(words);
		}
	}
	void compile(){
		int num_characters = _char_ids.size();
		int num_types = 280;	// Unicode
		_indices_wr_c = new Indices**[_c_max + 1];
		_indices_wp_c = new Indices**[_c_max + 1];
		for(int i = 0;i <= _c_max;i++){
			_indices_wr_c[i] = new Indices*[num_characters];
			_indices_wp_c[i] = new Indices*[num_characters];
			for(int j = 0;j < num_characters;j++){
				_indices_wr_c[i][j] = new Indices();
				_indices_wp_c[i][j] = new Indices();
			}
		}
		_indices_wr_t = new Indices**[_t_max + 1];
		_indices_wp_t = new Indices**[_t_max + 1];
		for(int i = 0;i <= _t_max;i++){
			_indices_wr_t[i] = new Indices*[num_types];
			_indices_wp_t[i] = new Indices*[num_types];
			for(int j = 0;j < num_types;j++){
				_indices_wr_t[i][j] = new Indices();
				_indices_wp_t[i][j] = new Indices();
			}
		}
		_indices_wr_cont = new Indices*[_coverage - 1];
		_indices_wp_cont = new Indices*[_coverage - 1];
		_indices_wr_ch = new Indices*[_coverage - 1];
		_indices_wp_ch = new Indices*[_coverage - 1];
		for(int i = 0;i < _coverage - 1;i++){
			_indices_wr_cont[i] = new Indices();
			_indices_wp_cont[i] = new Indices();
			_indices_wr_ch[i] = new Indices();
			_indices_wp_ch[i] = new Indices();
		}
		std::pair<int, int*> pair;
		for(auto word: _word_set){
			int word_length = word.size();
			int* features = extract_features(word);
			pair.first = word_length;
			pair.second = features;
			_length_features_pair.push_back(pair);
			// 重みの変更の影響を受ける素性ベクトルをリストアップ
		}
		_glm->init_weights(_char_ids.size());
		_compiled = true;
	}
	void save(string filename){

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
		int num_features = _glm->get_num_features();
		for(auto word: _word_set){
			int* feature = extract_features(word);
			wcout << word << "	";
			for(int i = 0;i < num_features;i++){
				wcout << feature[i] << ", ";
			}
			double p = _glm->compute_p(feature);
			double r = _glm->compute_r(feature);
			wcout << p << "	" << r << "	" << endl;
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
	int get_character_id(wchar_t character){
		auto itr = _char_ids.find(character);
		if(itr == _char_ids.end()){
			return -1;
		}
		return itr->second;
	}
	int* extract_features(wstring &word){
		int num_features = _glm->get_num_features();
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
};

class PyGLM{
public:
	GLM* _glm;
	PyGLM(string filename){
		_glm = new GLM();
	}
	~PyGLM(){

	}
	void load(string filename){
		
	}
};

BOOST_PYTHON_MODULE(model){
	python::class_<PyTrainer>("trainer", python::init<int, int, int>())
	.def("add_textfile", &PyTrainer::add_textfile)
	.def("save", &PyTrainer::save);

	python::class_<PyGLM>("glm", python::init<std::string>())
	.def("load", &PyGLM::load);
}