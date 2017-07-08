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
public:
	vector<int> _indices;
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
	Indices*** _indices_wx_c;
	Indices*** _indices_wx_t;
	Indices** _indices_wx_cont;
	Indices** _indices_wx_ch;
public:
	GLM* _glm;
	unordered_set<wstring> _word_set;
	unordered_map<wchar_t, int> _char_ids;
	int _coverage;
	int _c_max;
	int _t_max;
	double _randwalk_sigma;

	PyTrainer(int coverage, int c_max, int t_max, double randwalk_sigma){
		_glm = new GLM(coverage, c_max, t_max);
		_coverage = coverage;
		_c_max = c_max;
		_t_max = t_max;
		_randwalk_sigma = randwalk_sigma;
		_compiled = false;
		_indices_wx_c = NULL;
		_indices_wx_t = NULL;
		_indices_wx_cont = NULL;
		_indices_wx_ch = NULL;
	}
	~PyTrainer(){
		for(auto pair: _length_features_pair){
			delete[] pair.second;
		}
		delete _glm;
		if(_compiled){
			int num_characters = _char_ids.size();
			int num_types = CTYPE_TOTAL_TYPE;	// Unicode
			for(int i = 0;i <= _c_max;i++){
				for(int j = 0;j < num_characters;j++){
					delete _indices_wx_c[i][j];
				}
				delete[] _indices_wx_c[i];
			}
			delete[] _indices_wx_c;
			for(int i = 0;i <= _t_max;i++){
				for(int j = 0;j < num_types;j++){
					delete _indices_wx_t[i][j];
				}
				delete[] _indices_wx_t[i];
			}
			delete[] _indices_wx_t;
			for(int i = 0;i < _coverage;i++){
				delete _indices_wx_cont[i];
				delete _indices_wx_ch[i];
			}
			delete[] _indices_wx_cont;
			delete[] _indices_wx_ch;
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
		int num_types = CTYPE_TOTAL_TYPE;	// Unicode
		_indices_wx_c = new Indices**[_c_max + 1];
		for(int i = 0;i <= _c_max;i++){
			_indices_wx_c[i] = new Indices*[num_characters];
			for(int j = 0;j < num_characters;j++){
				_indices_wx_c[i][j] = new Indices();
			}
		}
		_indices_wx_t = new Indices**[_t_max + 1];
		for(int i = 0;i <= _t_max;i++){
			_indices_wx_t[i] = new Indices*[num_types];
			for(int j = 0;j < num_types;j++){
				_indices_wx_t[i][j] = new Indices();
			}
		}
		_indices_wx_cont = new Indices*[_coverage];
		_indices_wx_ch = new Indices*[_coverage];
		for(int i = 0;i < _coverage;i++){
			_indices_wx_cont[i] = new Indices();
			_indices_wx_ch[i] = new Indices();
		}
		std::pair<int, int*> pair;
		for(auto word: _word_set){
			int word_length = word.size();
			int* features = extract_features(word);
			pair.first = word_length;
			pair.second = features;
			_length_features_pair.push_back(pair);
			int feature_index = _length_features_pair.size() - 1;
			// 重みの変更の影響を受ける素性ベクトルをリストアップ
			// wcout << "word: " << word << ", f: " << feature_index << ", feature: ";
			// int num_features = _glm->get_num_features();
			// for(int i = 0;i < num_features;i++){
			// 	wcout << features[i] << ", ";
			// }
			// wcout << endl;

			for(int i = 0;i <= _c_max;i++){
				int cid = features[i] - 1;	// 文字IDは1スタート
				if(cid == -1){
					continue;	// 訓練データに無い、または単語の文字数がc_max未満
				}
				assert(cid < num_characters);
				// cout << "cid: " << cid << endl;
				_indices_wx_c[i][cid]->add(feature_index);
			}
			for(int i = 0;i <= _t_max;i++){
				int type = features[i + _c_max + 1];
				assert(type < CTYPE_TOTAL_TYPE);
				// cout << "type: " << type << endl;
				_indices_wx_t[i][type]->add(feature_index);
			}
			int cont = features[_c_max + _t_max + 2];
			// cout << "cont: " << cont << endl;
			_indices_wx_cont[cont]->add(feature_index);
			int ch = features[_c_max + _t_max + 3];
			// cout << "ch: " << ch << endl;
			_indices_wx_ch[ch]->add(feature_index);

		}

		// cout << "character:" << endl;
		// for(int i = 0;i <= _c_max;i++){
		// 	cout << "	i=" << i << endl;
		// 	for(int j = 0;j < num_characters;j++){
		// 		cout << "		j=" << j << ", size=" << _indices_wx_c[i][j]->size() << endl;
		// 	}
		// }
		// cout << "type:" << endl;
		// for(int i = 0;i <= _t_max;i++){
		// 	cout << "	i=" << i << endl;
		// 	for(int j = 0;j < num_types;j++){
		// 		cout << "		j=" << j << ", size=" << _indices_wx_t[i][j]->size() << endl;
		// 	}
		// }
		// cout << "cont:" << endl;
		// for(int i = 0;i < _coverage;i++){
		// 	cout << "	i=" << i << ", size=" << _indices_wx_cont[i]->size() << endl;
		// }
		// cout << "ch:" << endl;
		// for(int i = 0;i < _coverage;i++){
		// 	cout << "	i=" << i << ", size=" << _indices_wx_ch[i]->size() << endl;
		// }
		_glm->init_weights(_char_ids.size());
		_compiled = true;
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
	int* extract_features(const wstring &word){
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
	double compute_joint_log_likelihood_given_indices(const vector<int> &indices){
		double ll = 0;
		for(int i: indices){
			std::pair<int, int*> &pair = _length_features_pair[i];
			int word_length = pair.first;
			int* features = pair.second;
			double r = _glm->compute_r(features);
			double p = _glm->compute_p(features);
			ll += _glm->compute_nb_log_likelihood(word_length, r, p);
		}
		return ll;
	}
	double compute_joint_log_likelihood(){
		double ll = 0;
		for(auto &pair: _length_features_pair){
			int word_length = pair.first;
			int* features = pair.second;
			double r = _glm->compute_r(features);
			double p = _glm->compute_p(features);
			ll += _glm->compute_nb_log_likelihood(word_length, r, p);
		}
		return ll;
	}
	void perform_mcmc(){
		sample_wp_c_randomly();
		sample_wp_t_randomly();
		sample_wp_cont_randomly();
		sample_wp_ch_randomly();
		sample_wr_c_randomly();
		sample_wr_t_randomly();
		sample_wr_cont_randomly();
		sample_wr_ch_randomly();
	}
	void sample_wp_c_randomly(){
		int num_characters = _char_ids.size();
		for(int i = 0;i <= _c_max;i++){
			int cid = sampler::randint(1, num_characters + 1);	// 文字IDは1スタート
			Indices* indices = _indices_wx_c[i][cid - 1];			// 配列は0から
			if(indices->size() == 0){
				continue;
			}
			// cout << "#indices: " << indices->size() << endl;
			double old_weight = _glm->_wp_c[i][cid];
			double new_weight = old_weight + sampler::normal(0, _randwalk_sigma);
			double ll_old = compute_joint_log_likelihood_given_indices(indices->_indices);
			_glm->_wp_c[i][cid] = new_weight;
			double ll_new = compute_joint_log_likelihood_given_indices(indices->_indices);
			// cout << "before: " << ll_old << ", after: " << ll_new << endl;
			double acceptance_ratio = std::min(exp(ll_new - ll_old), 1.0);
			// cout << "acceptance_ratio: " << acceptance_ratio << endl;
			double bernoulli = sampler::uniform(0, 1);
			if(bernoulli > acceptance_ratio){
				_glm->_wp_c[i][cid] = old_weight;	// 棄却
			}
		}
	}
	void sample_wp_t_randomly(){
		int num_types = CTYPE_TOTAL_TYPE;	// Unicode
		for(int i = 0;i <= _t_max;i++){
			int type = sampler::randint(0, num_types);
			Indices* indices = _indices_wx_t[i][type];
			if(indices->size() == 0){
				continue;
			}
			// cout << "#indices: " << indices->size() << endl;
			double old_weight = _glm->_wp_t[i][type];
			double new_weight = old_weight + sampler::normal(0, _randwalk_sigma);
			double ll_old = compute_joint_log_likelihood_given_indices(indices->_indices);
			_glm->_wp_t[i][type] = new_weight;
			double ll_new = compute_joint_log_likelihood_given_indices(indices->_indices);
			// cout << "before: " << ll_old << ", after: " << ll_new << endl;
			double acceptance_ratio = std::min(exp(ll_new - ll_old), 1.0);
			// cout << "acceptance_ratio: " << acceptance_ratio << endl;
			double bernoulli = sampler::uniform(0, 1);
			if(bernoulli > acceptance_ratio){
				_glm->_wp_t[i][type] = old_weight;	// 棄却
			}
		}
	}
	void sample_wp_cont_randomly(){
		int cont = sampler::randint(0, _coverage);
		Indices* indices = _indices_wx_cont[cont];
		if(indices->size() == 0){
			return;
		}
		// cout << "#indices: " << indices->size() << endl;
		double old_weight = _glm->_wp_cont[cont];
		double new_weight = old_weight + sampler::normal(0, _randwalk_sigma);
		double ll_old = compute_joint_log_likelihood_given_indices(indices->_indices);
		_glm->_wp_cont[cont] = new_weight;
		double ll_new = compute_joint_log_likelihood_given_indices(indices->_indices);
		// cout << "before: " << ll_old << ", after: " << ll_new << endl;
		double acceptance_ratio = std::min(exp(ll_new - ll_old), 1.0);
		// cout << "acceptance_ratio: " << acceptance_ratio << endl;
		double bernoulli = sampler::uniform(0, 1);
		if(bernoulli > acceptance_ratio){
			_glm->_wp_cont[cont] = old_weight;	// 棄却
		}
	}
	void sample_wp_ch_randomly(){
		int ch = sampler::randint(0, _coverage);
		Indices* indices = _indices_wx_ch[ch];
		if(indices->size() == 0){
			return;
		}
		// cout << "#indices: " << indices->size() << endl;
		double old_weight = _glm->_wp_ch[ch];
		double new_weight = old_weight + sampler::normal(0, _randwalk_sigma);
		double ll_old = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(old_weight);
		_glm->_wp_ch[ch] = new_weight;
		double ll_new = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(new_weight);
		// cout << "before: " << ll_old << ", after: " << ll_new << endl;
		double acceptance_ratio = std::min(exp(ll_new - ll_old), 1.0);
		// cout << "acceptance_ratio: " << acceptance_ratio << endl;
		double bernoulli = sampler::uniform(0, 1);
		if(bernoulli > acceptance_ratio){
			_glm->_wp_ch[ch] = old_weight;	// 棄却
		}
	}

	void sample_wr_c_randomly(){
		int num_characters = _char_ids.size();
		for(int i = 0;i <= _c_max;i++){
			int cid = sampler::randint(1, num_characters + 1);	// 文字IDは1スタート
			Indices* indices = _indices_wx_c[i][cid - 1];			// 配列は0から
			if(indices->size() == 0){
				continue;
			}
			// cout << "#indices: " << indices->size() << endl;
			double old_weight = _glm->_wr_c[i][cid];
			double new_weight = old_weight + sampler::normal(0, _randwalk_sigma);
			double ll_old = compute_joint_log_likelihood_given_indices(indices->_indices);
			_glm->_wr_c[i][cid] = new_weight;
			double ll_new = compute_joint_log_likelihood_given_indices(indices->_indices);
			// cout << "before: " << ll_old << ", after: " << ll_new << endl;
			double acceptance_ratio = std::min(exp(ll_new - ll_old), 1.0);
			// cout << "acceptance_ratio: " << acceptance_ratio << endl;
			double bernoulli = sampler::uniform(0, 1);
			if(bernoulli > acceptance_ratio){
				_glm->_wr_c[i][cid] = old_weight;	// 棄却
			}
		}
	}
	void sample_wr_t_randomly(){
		int num_types = CTYPE_TOTAL_TYPE;	// Unicode
		for(int i = 0;i <= _t_max;i++){
			int type = sampler::randint(0, num_types);
			Indices* indices = _indices_wx_t[i][type];
			if(indices->size() == 0){
				continue;
			}
			// cout << "#indices: " << indices->size() << endl;
			double old_weight = _glm->_wr_t[i][type];
			double new_weight = old_weight + sampler::normal(0, _randwalk_sigma);
			double ll_old = compute_joint_log_likelihood_given_indices(indices->_indices);
			_glm->_wr_t[i][type] = new_weight;
			double ll_new = compute_joint_log_likelihood_given_indices(indices->_indices);
			// cout << "before: " << ll_old << ", after: " << ll_new << endl;
			double acceptance_ratio = std::min(exp(ll_new - ll_old), 1.0);
			// cout << "acceptance_ratio: " << acceptance_ratio << endl;
			double bernoulli = sampler::uniform(0, 1);
			if(bernoulli > acceptance_ratio){
				_glm->_wr_t[i][type] = old_weight;	// 棄却
			}
		}
	}
	void sample_wr_cont_randomly(){
		int cont = sampler::randint(0, _coverage);
		Indices* indices = _indices_wx_cont[cont];
		if(indices->size() == 0){
			return;
		}
		// cout << "#indices: " << indices->size() << endl;
		double old_weight = _glm->_wr_cont[cont];
		double new_weight = old_weight + sampler::normal(0, _randwalk_sigma);
		double ll_old = compute_joint_log_likelihood_given_indices(indices->_indices);
		_glm->_wr_cont[cont] = new_weight;
		double ll_new = compute_joint_log_likelihood_given_indices(indices->_indices);
		// cout << "before: " << ll_old << ", after: " << ll_new << endl;
		double acceptance_ratio = std::min(exp(ll_new - ll_old), 1.0);
		// cout << "acceptance_ratio: " << acceptance_ratio << endl;
		double bernoulli = sampler::uniform(0, 1);
		if(bernoulli > acceptance_ratio){
			_glm->_wr_cont[cont] = old_weight;	// 棄却
		}
	}
	void sample_wr_ch_randomly(){
		int ch = sampler::randint(0, _coverage);
		Indices* indices = _indices_wx_ch[ch];
		if(indices->size() == 0){
			return;
		}
		// cout << "#indices: " << indices->size() << endl;
		double old_weight = _glm->_wr_ch[ch];
		double new_weight = old_weight + sampler::normal(0, _randwalk_sigma);
		double ll_old = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(old_weight);
		_glm->_wr_ch[ch] = new_weight;
		double ll_new = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(new_weight);
		// cout << "before: " << ll_old << ", after: " << ll_new << endl;
		double acceptance_ratio = std::min(exp(ll_new - ll_old), 1.0);
		// cout << "acceptance_ratio: " << acceptance_ratio << endl;
		double bernoulli = sampler::uniform(0, 1);
		if(bernoulli > acceptance_ratio){
			_glm->_wr_ch[ch] = old_weight;	// 棄却
		}
	}
	void save(string filename){
		_glm->save(filename);
	}
	void load(string filename){
		_glm->load(filename);
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
	python::class_<PyTrainer>("trainer", python::init<int, int, int, double>())
	.def("add_textfile", &PyTrainer::add_textfile)
	.def("compile", &PyTrainer::compile)
	.def("perform_mcmc", &PyTrainer::perform_mcmc)
	.def("compute_joint_log_likelihood", &PyTrainer::compute_joint_log_likelihood)
	.def("save", &PyTrainer::save);

	python::class_<PyGLM>("glm", python::init<std::string>())
	.def("load", &PyGLM::load);
}