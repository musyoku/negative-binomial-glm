#include <numeric>
#include "../model.cpp"
using namespace std;

int main(int argc, char *argv[]){
	PyGLM* glm = new PyGLM();
	assert(glm->load("glm.model"));
	string filename = "../../../dataset/test.txt";
	wifstream ifs(filename.c_str());
    // ifs.imbue(locale(locale::empty(), new codecvt_utf8<wchar_t>));
	double threshold = 0.99;
	int max_word_length = 16;
	assert(ifs.fail() == false);

	// 統計
	vector<double> errors;
	vector<int> pred_length(max_word_length, 0);
	vector<int> atari(max_word_length, 0);
	vector<int> total(max_word_length, 0);
	int total_global = 0;
	int atari_global = 0;

	wstring sentence;
	while (getline(ifs, sentence)){
		if(sentence.empty()){
			continue;
		}
		vector<wstring> words;
		split_word_by(sentence, L' ', words);
		for(auto &word: words){
			int length_pred = glm->predict_word_length(word, threshold, max_word_length);
			assert(length_pred <= max_word_length);
			if(length_pred >= word.length()){
				errors.push_back(length_pred - word.length());
				atari[word.length() - 1] += 1;
				atari_global += 1;
				wcout << word << " pred: " << length_pred << " - actual: " << word.length() << " " << word.length() << endl;
			}
			total[word.length() - 1] += 1;
			pred_length[length_pred - 1] += 1;
			total_global += 1;
		}
	}
	cout << "\e[1mL:Precision: \e[0m" << endl;
	for(int l = 0;l < max_word_length;l++){
		double precision = atari[l] / (double)total[l];
		cout << l + 1 << ":	" << precision << endl;
	}
	cout << "total: " << atari_global / (double)total_global << endl;

	cout << "\e[1mL: Frequency\e[0m" << endl;
	for(int l = 0;l < max_word_length;l++){
		cout << l + 1 << ":	" << pred_length[l] << endl;
	}

	// 予測と正解の誤差の平均・分散
	double sum = std::accumulate(errors.begin(), errors.end(), 0.0);
	double mean = sum / errors.size();
	vector<double> diff(errors.size());
	std::transform(errors.begin(), errors.end(), diff.begin(), [mean](double x) { return x - mean; });
	double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
	double std = std::sqrt(sq_sum / errors.size());
	cout << "\e[1mError:\e[0m" << endl;
	cout << "mean: " << mean << " - stddev: " << std << endl;

	delete glm;
}
