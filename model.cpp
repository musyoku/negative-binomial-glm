#include <boost/python.hpp>
#include "src/glm.h"
using namespace boost;
using namespace npycrf;

class PyTrainer{
public:
	GLM* _glm;
	PyTrainer(int coverage){
		_glm = new GLM(coverage);
	}
	~PyTrainer(){

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
	python::class_<PyTrainer>("trainer", python::init<int>())
	.def("save", &PyTrainer::save);

	python::class_<PyGLM>("glm", python::init<std::string>())
	.def("load", &PyGLM::load);
}