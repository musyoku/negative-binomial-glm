#include "model.cpp"
using namespace std;

int main(int argc, char *argv[]){
	// 日本語周り
	setlocale(LC_CTYPE, "");
	ios_base::sync_with_stdio(false);
	locale default_loc("");
	locale::global(default_loc);
	locale ctype_default(locale::classic(), default_loc, locale::ctype);
	wcout.imbue(ctype_default);
	wcin.imbue(ctype_default);

	PyTrainer* trainer = new PyTrainer(8, 1, 4, 0.1);
	trainer->add_textfile("../../dataset/japanese.txt");
	trainer->compile();
	// trainer->dump_words();
	for(int i = 0;i < 1000;i++){
		trainer->perform_mcmc();
		double ll = trainer->compute_joint_log_likelihood();
		cout << ll << endl;
	}
	delete trainer;
}
