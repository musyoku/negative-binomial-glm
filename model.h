#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/functional/hash.hpp>
#include <locale>
#include <fstream>
#include <vector>
#include <cassert>
#include "src/glm.h"

void split_word_by(const std::wstring &str, wchar_t delim, std::vector<std::wstring> &words){
	words.clear();
	std::wstring item;
	for(wchar_t ch: str){
		if (ch == delim){
			if (!item.empty() && item.length() > 0){
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
	std::vector<int> _indices;
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

class GLM{
public:
	npycrf::GLM* _glm;
	std::unordered_map<wchar_t, int> _char_ids;
	GLM(){
		setlocale(LC_CTYPE, "");
		std::ios_base::sync_with_stdio(false);
		std::locale default_loc("");
		std::locale::global(default_loc);
		std::locale ctype_default(std::locale::classic(), default_loc, std::locale::ctype);
		std::wcout.imbue(ctype_default);
		std::wcin.imbue(ctype_default);
		_glm = NULL;
	}
	GLM(std::string filename){
		setlocale(LC_CTYPE, "");
		std::ios_base::sync_with_stdio(false);
		std::locale default_loc("");
		std::locale::global(default_loc);
		std::locale ctype_default(std::locale::classic(), default_loc, std::locale::ctype);
		std::wcout.imbue(ctype_default);
		std::wcin.imbue(ctype_default);
		_glm = NULL;
		load(filename);
	}
	~GLM(){
		delete _glm;
	}
	int coverage(){
		if(_glm == NULL){
			return -1;
		}
		return _glm->_coverage;
	}
	bool load(std::string filename){
		if(_glm == NULL){
			_glm = new npycrf::GLM();
		}
		std::ifstream ifs(filename);
		if(ifs.good()){
			_glm = new npycrf::GLM();
			boost::archive::binary_iarchive iarchive(ifs);
			iarchive >> *_glm;
			iarchive >> _char_ids;
			return true;
		}
		return false;
	}
	int get_character_id(wchar_t character){
		auto itr = _char_ids.find(character);
		if(itr == _char_ids.end()){
			return -1;
		}
		return itr->second;
	}
	int get_word_length_exceeds_threshold(double r, double p, double threshold, int max_word_length){
		int l = 1;
		for(;l <= max_word_length;l++){
			double cum = _glm->compute_cumulative_probability(l, r, p);
			if(cum >= threshold){
				break;
			}
		}
		return std::min(l, max_word_length);
	}
	int predict_word_length(std::wstring word, double threshold, int max_word_length){
		int* feature = extract_features(word);
		double p = _glm->compute_p(feature);
		double r = _glm->compute_r(feature);
		int length = get_word_length_exceeds_threshold(r, p, threshold, max_word_length);
		assert(length <= max_word_length);
		delete[] feature;
		return length;
	}
	int* extract_features(const std::wstring &word){
		int c_max = _glm->_c_max;
		int t_max = _glm->_t_max;
		int coverage = _glm->_coverage;
		int num_features = _glm->get_num_features();
		int* features = new int[num_features];
		for(int i = 0;i < num_features;i++){
			features[i] = 0;
		}
		wchar_t character;
		int char_id;
		int t = word.size() - 1;
		// 文字による素性
		for(int i = 0;i <= c_max;i++){
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
		for(int i = 0;i <= t_max;i++){
			if(t - i < 0){
				break;
			}
			character = word[t - i];
			unsigned int type = chartype::get_type(character);
			features[i + c_max + 1] = type;
		}
		// t以前の同じ文字種の数
		int cont = 0;
		unsigned int basetype = chartype::get_type(word[t]);
		for(int i = 1;i < coverage;i++){
			if(t - i < 0){
				break;
			}
			character = word[t - i];
			unsigned int type = chartype::get_type(character);
			if(type == basetype){
				cont += 1;
			}
		}
		features[c_max + t_max + 2] = cont;
		// 文字種が変わった数
		int ch = 0;
		for(int i = 1;i < coverage;i++){
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
		features[c_max + t_max + 3] = ch;
		return features;
	}
};

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
		std::size_t seed = 0;
		boost::hash_combine(seed, p.first);
		boost::hash_combine(seed, p.second);
		return seed;
    }
};

class GLMTrainer: GLM{
private:
	std::vector<std::pair<int, int*>> _length_features_pair;
	bool _compiled;
	// 重みの変更の影響を受ける素性ベクトルをリストアップ
	Indices*** _indices_wx_c;
	Indices*** _indices_wx_t;
	Indices** _indices_wx_cont;
	Indices** _indices_wx_ch;
public:
	std::unordered_set<std::pair<int, std::wstring>, pair_hash> _length_substr_set;
	int _coverage;
	int _c_max;
	int _t_max;
	double _randwalk_sigma;
	int _mcmc_total_transition;
	int _mcmc_num_rejection;

	GLMTrainer(int coverage, int c_max, int t_max, double randwalk_sigma){
		// 日本語周り
		setlocale(LC_CTYPE, "");
		std::ios_base::sync_with_stdio(false);
		std::locale default_loc("");
		std::locale::global(default_loc);
		std::locale ctype_default(std::locale::classic(), default_loc, std::locale::ctype);
		std::wcout.imbue(ctype_default);
		std::wcin.imbue(ctype_default);

		_glm = new npycrf::GLM(coverage, c_max, t_max);
		_coverage = coverage;
		_c_max = c_max;
		_t_max = t_max;
		_randwalk_sigma = randwalk_sigma;
		_compiled = false;
		_indices_wx_c = NULL;
		_indices_wx_t = NULL;
		_indices_wx_cont = NULL;
		_indices_wx_ch = NULL;
		_mcmc_total_transition = 0;
		_mcmc_num_rejection = 0;
	}
	~GLMTrainer(){
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
	void add_textfile(std::string filename){
		std::wifstream ifs(filename.c_str());
		std::wstring sentence;
		assert(ifs.fail() == false);
		while (getline(ifs, sentence)){
			if(sentence.empty()){
				continue;
			}
			std::vector<std::wstring> words;
			split_word_by(sentence, L' ', words);
			add_words(words);
		}
	}
	void compile(){
		assert(_compiled == false);
		int num_characters = _char_ids.size();
		assert(num_characters > 0);
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
		for(auto &data: _length_substr_set){
			int true_length = data.first;
			int* features = extract_features(data.second);
			pair.first = true_length;
			pair.second = features;
			_length_features_pair.push_back(pair);
			int feature_index = _length_features_pair.size() - 1;
			// 重みの変更の影響を受ける素性ベクトルをリストアップ
			for(int i = 0;i <= _c_max;i++){
				int cid = features[i] - 1;	// 文字IDは1スタート
				if(cid == -1){
					continue;	// 訓練データに無い、または単語の文字数がc_max未満
				}
				assert(cid < num_characters);
				// std::cout << "cid: " << cid << std::endl;
				_indices_wx_c[i][cid]->add(feature_index);
			}
			for(int i = 0;i <= _t_max;i++){
				int type = features[i + _c_max + 1];
				assert(type < CTYPE_TOTAL_TYPE);
				// std::cout << "type: " << type << std::endl;
				_indices_wx_t[i][type]->add(feature_index);
			}
			int cont = features[_c_max + _t_max + 2];
			// std::cout << "cont: " << cont << std::endl;
			_indices_wx_cont[cont]->add(feature_index);
			int ch = features[_c_max + _t_max + 3];
			// std::cout << "ch: " << ch << std::endl;
			_indices_wx_ch[ch]->add(feature_index);

		}
		_glm->init_weights(_char_ids.size());
		_compiled = true;
	}
	void add_words(const std::vector<std::wstring> &words){
		std::pair<int, std::wstring> pair;
		// 文にする
		std::wstring sentence;
		for(const auto &word: words){
			sentence += word;
		}
		// std::wcout << sentence << std::endl;
		int substr_end = -1;
		for(const auto &word: words){
			substr_end += word.length();
			// 単語ではなく単語を含む部分文字列にする
			int substr_start = std::max(0, substr_end - _coverage + 1);	// coverageの範囲の文字列を全て取る
			std::wstring substr(sentence.begin() + substr_start, sentence.begin() + substr_end + 1);
			// std::wcout << word << " : " << substr << std::endl;
			int true_length = word.length();
			pair.first = true_length;
			pair.second = substr;
			auto itr = _length_substr_set.find(pair);
			if(itr == _length_substr_set.end()){
				_length_substr_set.insert(pair);
			}
			add_character(word);
		}
	}
	void add_character(const std::wstring &word){
		for(auto character: word){
			auto itr = _char_ids.find(character);
			if(itr == _char_ids.end()){
				int cid = _char_ids.size() + 1;
				_char_ids[character] = cid;    // 0避け
			}
		}
	}
	void dump_words(){
		std::cout << "word	feature" << std::endl;
		int num_features = _glm->get_num_features();
		for(auto &pair: _length_substr_set){
			int true_length = pair.first;
			std::wstring substr = pair.second;
			int* feature = extract_features(substr);
			std::wcout << substr << "	";
			for(int i = 0;i < num_features;i++){
				std::wcout << feature[i] << ", ";
			}
			double p = _glm->compute_p(feature);
			double r = _glm->compute_r(feature);
			std::wcout << p << "	" << r << "	" << std::endl;
			delete[] feature;
		}
	}
	void dump_characters(){
		std::cout << "char	id	type" << std::endl;
		for(auto elem: _char_ids){
			wchar_t character = elem.first;
			unsigned int type = chartype::get_type(character);
			std::wstring_convert<std::codecvt_utf8<wchar_t>,wchar_t> cv;
			std::wstring name = cv.from_bytes(chartype::get_name(type));
			std::wcout << character << "	" << elem.second << "	" << name << std::endl;
		}
	}
	double compute_joint_log_likelihood_given_indices(const std::vector<int> &indices){
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
			int cid = npycrf::sampler::randint(1, num_characters + 1);	// 文字IDは1スタート
			Indices* indices = _indices_wx_c[i][cid - 1];			// 配列は0から
			if(indices->size() == 0){
				continue;
			}
			// std::cout << "#indices: " << indices->size() << std::endl;
			double old_weight = _glm->_wp_c[i][cid];
			double new_weight = old_weight + npycrf::sampler::normal(0, _randwalk_sigma);
			double ll_old = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(old_weight);
			_glm->_wp_c[i][cid] = new_weight;
			double ll_new = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(new_weight);
			// std::cout << "before: " << ll_old << ", after: " << ll_new << std::endl;
			double acceptance_ratio = std::min(exp(ll_new - ll_old), 1.0);
			// std::cout << "acceptance_ratio: " << acceptance_ratio << std::endl;
			double bernoulli = npycrf::sampler::uniform(0, 1);
			_mcmc_total_transition += 1;
			if(bernoulli > acceptance_ratio){
				_glm->_wp_c[i][cid] = old_weight;	// 棄却
				_mcmc_num_rejection += 1;
			}
		}
	}
	void sample_wp_t_randomly(){
		int num_types = CTYPE_TOTAL_TYPE;	// Unicode
		for(int i = 0;i <= _t_max;i++){
			int type = npycrf::sampler::randint(0, num_types);
			Indices* indices = _indices_wx_t[i][type];
			if(indices->size() == 0){
				continue;
			}
			// std::cout << "#indices: " << indices->size() << std::endl;
			double old_weight = _glm->_wp_t[i][type];
			double new_weight = old_weight + npycrf::sampler::normal(0, _randwalk_sigma);
			double ll_old = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(old_weight);
			_glm->_wp_t[i][type] = new_weight;
			double ll_new = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(new_weight);
			// std::cout << "before: " << ll_old << ", after: " << ll_new << std::endl;
			double acceptance_ratio = std::min(exp(ll_new - ll_old), 1.0);
			// std::cout << "acceptance_ratio: " << acceptance_ratio << std::endl;
			double bernoulli = npycrf::sampler::uniform(0, 1);
			_mcmc_total_transition += 1;
			if(bernoulli > acceptance_ratio){
				_glm->_wp_t[i][type] = old_weight;	// 棄却
				_mcmc_num_rejection += 1;
			}
		}
	}
	void sample_wp_cont_randomly(){
		int cont = npycrf::sampler::randint(0, _coverage);
		Indices* indices = _indices_wx_cont[cont];
		if(indices->size() == 0){
			return;
		}
		// std::cout << "#indices: " << indices->size() << std::endl;
		double old_weight = _glm->_wp_cont[cont];
		double new_weight = old_weight + npycrf::sampler::normal(0, _randwalk_sigma);
		double ll_old = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(old_weight);
		_glm->_wp_cont[cont] = new_weight;
		double ll_new = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(new_weight);
		// std::cout << "before: " << ll_old << ", after: " << ll_new << std::endl;
		double acceptance_ratio = std::min(exp(ll_new - ll_old), 1.0);
		// std::cout << "acceptance_ratio: " << acceptance_ratio << std::endl;
		double bernoulli = npycrf::sampler::uniform(0, 1);
		_mcmc_total_transition += 1;
		if(bernoulli > acceptance_ratio){
			_glm->_wp_cont[cont] = old_weight;	// 棄却
			_mcmc_num_rejection += 1;
		}
	}
	void sample_wp_ch_randomly(){
		int ch = npycrf::sampler::randint(0, _coverage);
		Indices* indices = _indices_wx_ch[ch];
		if(indices->size() == 0){
			return;
		}
		// std::cout << "#indices: " << indices->size() << std::endl;
		double old_weight = _glm->_wp_ch[ch];
		double new_weight = old_weight + npycrf::sampler::normal(0, _randwalk_sigma);
		double ll_old = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(old_weight);
		_glm->_wp_ch[ch] = new_weight;
		double ll_new = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(new_weight);
		// std::cout << "before: " << ll_old << ", after: " << ll_new << std::endl;
		double acceptance_ratio = std::min(exp(ll_new - ll_old), 1.0);
		// std::cout << "acceptance_ratio: " << acceptance_ratio << std::endl;
		double bernoulli = npycrf::sampler::uniform(0, 1);
		_mcmc_total_transition += 1;
		if(bernoulli > acceptance_ratio){
			_glm->_wp_ch[ch] = old_weight;	// 棄却
			_mcmc_num_rejection += 1;
		}
	}

	void sample_wr_c_randomly(){
		int num_characters = _char_ids.size();
		for(int i = 0;i <= _c_max;i++){
			int cid = npycrf::sampler::randint(1, num_characters + 1);	// 文字IDは1スタート
			Indices* indices = _indices_wx_c[i][cid - 1];			// 配列は0から
			if(indices->size() == 0){
				continue;
			}
			// std::cout << "#indices: " << indices->size() << std::endl;
			double old_weight = _glm->_wr_c[i][cid];
			double new_weight = old_weight + npycrf::sampler::normal(0, _randwalk_sigma);
			double ll_old = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(old_weight);
			_glm->_wr_c[i][cid] = new_weight;
			double ll_new = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(new_weight);
			// std::cout << "before: " << ll_old << ", after: " << ll_new << std::endl;
			double acceptance_ratio = std::min(exp(ll_new - ll_old), 1.0);
			// std::cout << "acceptance_ratio: " << acceptance_ratio << std::endl;
			double bernoulli = npycrf::sampler::uniform(0, 1);
			_mcmc_total_transition += 1;
			if(bernoulli > acceptance_ratio){
				_glm->_wr_c[i][cid] = old_weight;	// 棄却
				_mcmc_num_rejection += 1;
			}
		}
	}
	void sample_wr_t_randomly(){
		int num_types = CTYPE_TOTAL_TYPE;	// Unicode
		for(int i = 0;i <= _t_max;i++){
			int type = npycrf::sampler::randint(0, num_types);
			Indices* indices = _indices_wx_t[i][type];
			if(indices->size() == 0){
				continue;
			}
			// std::cout << "#indices: " << indices->size() << std::endl;
			double old_weight = _glm->_wr_t[i][type];
			double new_weight = old_weight + npycrf::sampler::normal(0, _randwalk_sigma);
			double ll_old = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(old_weight);
			_glm->_wr_t[i][type] = new_weight;
			double ll_new = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(new_weight);
			// std::cout << "before: " << ll_old << ", after: " << ll_new << std::endl;
			double acceptance_ratio = std::min(exp(ll_new - ll_old), 1.0);
			// std::cout << "acceptance_ratio: " << acceptance_ratio << std::endl;
			double bernoulli = npycrf::sampler::uniform(0, 1);
			if(bernoulli > acceptance_ratio){
				_glm->_wr_t[i][type] = old_weight;	// 棄却
				_mcmc_num_rejection += 1;
			}
		}
	}
	void sample_wr_cont_randomly(){
		int cont = npycrf::sampler::randint(0, _coverage);
		Indices* indices = _indices_wx_cont[cont];
		if(indices->size() == 0){
			return;
		}
		// std::cout << "#indices: " << indices->size() << std::endl;
		double old_weight = _glm->_wr_cont[cont];
		double new_weight = old_weight + npycrf::sampler::normal(0, _randwalk_sigma);
		double ll_old = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(old_weight);
		_glm->_wr_cont[cont] = new_weight;
		double ll_new = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(new_weight);
		// std::cout << "before: " << ll_old << ", after: " << ll_new << std::endl;
		double acceptance_ratio = std::min(exp(ll_new - ll_old), 1.0);
		// std::cout << "acceptance_ratio: " << acceptance_ratio << std::endl;
		double bernoulli = npycrf::sampler::uniform(0, 1);
		_mcmc_total_transition += 1;
		if(bernoulli > acceptance_ratio){
			_glm->_wr_cont[cont] = old_weight;	// 棄却
			_mcmc_num_rejection += 1;
		}
	}
	void sample_wr_ch_randomly(){
		int ch = npycrf::sampler::randint(0, _coverage);
		Indices* indices = _indices_wx_ch[ch];
		if(indices->size() == 0){
			return;
		}
		// std::cout << "#indices: " << indices->size() << std::endl;
		double old_weight = _glm->_wr_ch[ch];
		double new_weight = old_weight + npycrf::sampler::normal(0, _randwalk_sigma);
		double ll_old = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(old_weight);
		_glm->_wr_ch[ch] = new_weight;
		double ll_new = compute_joint_log_likelihood_given_indices(indices->_indices) + _glm->compute_log_weight_prior(new_weight);
		// std::cout << "before: " << ll_old << ", after: " << ll_new << std::endl;
		double acceptance_ratio = std::min(exp(ll_new - ll_old), 1.0);
		// std::cout << "acceptance_ratio: " << acceptance_ratio << std::endl;
		double bernoulli = npycrf::sampler::uniform(0, 1);
		_mcmc_total_transition += 1;
		if(bernoulli > acceptance_ratio){
			_glm->_wr_ch[ch] = old_weight;	// 棄却
			_mcmc_num_rejection += 1;
		}
	}
	int get_num_words(){
		return _length_substr_set.size();
	}
	int get_num_characters(){
		return _char_ids.size();
	}
	double compute_mean_precision(double threshold, int max_word_length){
		int total = 0;
		int atari = 0;
		for(auto &pair: _length_substr_set){
			int true_length = pair.first;
			int pred_length = predict_word_length(pair.second, threshold, max_word_length);
			if(pred_length >= true_length){
				atari += 1;
			}
			total += 1;
		}
		return atari / (double)total;
	}
	int predict_word_length(const std::wstring &word, double threshold, int max_word_length){
		int* feature = extract_features(word);
		double p = _glm->compute_p(feature);
		double r = _glm->compute_r(feature);
		int length = get_word_length_exceeds_threshold(r, p, threshold, max_word_length);
		delete[] feature;
		return length;
	}
	double get_acceptance_rate(){
		return 1.0 - (_mcmc_num_rejection + 1) / (double)(_mcmc_total_transition + 1);
	}
	void save(std::string filename){
		std::ofstream ofs(filename);
		boost::archive::binary_oarchive oarchive(ofs);
		oarchive << *_glm;
		oarchive << _char_ids;
	}
};