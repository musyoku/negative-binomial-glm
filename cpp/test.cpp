#include <numeric>
#include "../model.cpp"
using namespace std;

int main(int argc, char *argv[]){
	PyGLM* glm = new PyGLM();
	assert(glm->load("glm.model"));
	string filename = "../../../dataset/test.txt";
	wifstream ifs(filename.c_str());
    // ifs.imbue(locale(locale::empty(), new codecvt_utf8<wchar_t>));
	wstring sentence;
	int total = 0;
	int atari = 0;
	double threshold = 0.99;
	int max_word_length = 15;
	assert(ifs.fail() == false);

	// 統計
	vector<double> errors;
	vector<int> preds(max_word_length, 0);

	while (getline(ifs, sentence)){
		if(sentence.empty()){
			continue;
		}
		vector<wstring> words;
		split_word_by(sentence, L' ', words);
		for(auto &word: words){
			int length_pred = glm->predict_word_length(word, threshold, max_word_length);
			if(length_pred >= word.size()){
				atari += 1;
				errors.push_back(length_pred - word.size());
				// wcout << word << " pred: " << length_pred << " - actual: " << word.size() << " " << word.length() << endl;
			}
			preds[length_pred - 1] += 1;
			total += 1;
		}
	}
	cout << "precision: " << atari / (double)total * 100.0 << endl;
	cout << "L: Frequency" << endl;
	for(int l = 0;l < max_word_length;l++){
		cout << l + 1 << ":" << preds[l] << endl;
	}

	// 予測と正解の誤差の平均・分散
	double sum = std::accumulate(errors.begin(), errors.end(), 0.0);
	double mean = sum / errors.size();
	vector<double> diff(errors.size());
	std::transform(errors.begin(), errors.end(), diff.begin(), [mean](double x) { return x - mean; });
	double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
	double std = std::sqrt(sq_sum / errors.size());
	cout << "error mean: " << mean << " - stddev: " << std << endl;

	delete glm;
}
