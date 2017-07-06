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

	for(int i = 0;i < 10;i++){
		PyTrainer* trainer = new PyTrainer(8, 1, 1);
		trainer->add_textfile("../beluga-cstm/dataset/admin.txt");
		trainer->compile();
		delete trainer;
	}
}
