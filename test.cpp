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
	cout << sizeof(int) << endl;
	cout << sizeof(double) << endl;
	cout << sizeof(wchar_t) << endl;

}
