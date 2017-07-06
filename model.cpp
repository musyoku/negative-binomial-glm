#include <boost/python.hpp>
#include <fstream>
#include "src/glm.h"
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

class PyTrainer{
public:
	GLM* _glm;
	PyTrainer(int coverage, int c_max, int t_max){
		setlocale(LC_CTYPE, "ja_JP.UTF-8");
		ios_base::sync_with_stdio(false);
		locale default_loc("ja_JP.UTF-8");
		locale::global(default_loc);
		locale ctype_default(locale::classic(), default_loc, locale::ctype); //※
		wcout.imbue(ctype_default);
		wcin.imbue(ctype_default);
		_glm = new GLM(coverage, c_max, t_max);
	}
	~PyTrainer(){

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
			_glm->add_words(words);
		}
		// cout << _glm->get_num_features() << endl;
		_glm->dump_characters();
		_glm->dump_words();
	}
	void save(string filename){

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