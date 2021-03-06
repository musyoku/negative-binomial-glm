#include <numeric>
#include "model.h"

using namespace std;

double compute_mean(vector<double> &v){
	double sum = accumulate(v.begin(), v.end(), 0.0);
	double mean = sum / v.size();
	return mean;
}

double compute_stddev(vector<double> &v, double mean){
	vector<double> diff(v.size());
	transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
	double sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
	double stddev = sqrt(sq_sum / v.size());
	return stddev;
}

int main(int argc, char *argv[]){
	GLM* glm = new GLM();
	assert(glm->load("glm.model"));
	string filename = "../../dataset/test.txt";
	wifstream ifs(filename.c_str());
	double threshold = 0.99;
	int max_word_length = 16;
	assert(ifs.fail() == false);

	// 統計
	vector<vector<double>> errors(max_word_length);
	vector<int> histogram_pred_length(max_word_length, 0);
	vector<int> histogram_true_length(max_word_length, 0);
	vector<int> histogram_atari(max_word_length, 0);
	vector<int> histogram_attempt(max_word_length, 0);
	vector<int> histogram_atari_over_5(max_word_length, 0);
	vector<int> histogram_attempt_over_5(max_word_length, 0);
	unordered_set<wstring> word_set;
	int total_attempt_count = 0;
	int total_atari_count = 0;
	int total_attempt_over_5_count = 0;
	int total_atari_over_5_count = 0;

	wstring sentence;
	while (getline(ifs, sentence)){
		if(sentence.empty()){
			continue;
		}
		vector<wstring> words;
		split_word_by(sentence, L' ', words);

		// 文にする
		wstring string;
		for(const auto &word: words){
			string += word;
		}
		int substr_end = -1;
		for(auto &word: words){
			substr_end += word.length();
			// 単語ではなく単語を含む部分文字列にする
			int substr_start = std::max(0, substr_end - glm->coverage());	// coverageの範囲の文字列を全て取る
			wstring substr(string.begin() + substr_start, string.begin() + substr_end + 1);
			int true_length = word.length();	// 真の長さは単語の長さ
			if(true_length > max_word_length){
				continue;
			}
			int pred_length = glm->predict_word_length(substr, threshold, max_word_length);
			assert(pred_length <= max_word_length);
			if(pred_length >= true_length){
				histogram_atari[true_length - 1] += 1;
				if(true_length >= 5){
					total_atari_over_5_count += 1;
				}
				total_atari_count += 1;
				// wcout << substr << " pred: " << pred_length << " - actual: " << true_length << " " << true_length << endl;
			}
			errors[true_length - 1].push_back(pred_length - true_length);
			histogram_attempt[true_length - 1] += 1;
			if(true_length >= 5){
				total_attempt_over_5_count += 1;
			}
			histogram_pred_length[pred_length - 1] += 1;
			word_set.insert(word);
			total_attempt_count += 1;
		}
	}
	cout << "\e[1mL	Precision \e[0m" << endl;
	for(int l = 0;l < max_word_length;l++){
		double precision = histogram_atari[l] / (double)histogram_attempt[l];
		cout << l + 1 << ":	" << precision << endl;
	}
	cout << "n ≥ 5:	" << total_atari_over_5_count / (double)total_attempt_over_5_count << endl;
	cout << "all:	" << total_atari_count / (double)total_attempt_count << endl;

	cout << "\e[1mDistribution of predicted maximum word lengths:\e[0m" << endl;
	cout << "\e[1mL	Frequency\e[0m" << endl;
	for(int l = 0;l < max_word_length;l++){
		cout << l + 1 << ":	" << histogram_pred_length[l] << endl;
	}

	cout << "\e[1mDistribution of true word lengths:\e[0m" << endl;
	cout << "\e[1mL	Frequency\e[0m" << endl;
	for(const auto &word: word_set){
		int length = word.length();
		histogram_true_length[length - 1] += 1;
	}
	for(int l = 0;l < max_word_length;l++){
		cout << l + 1 << ":	" << histogram_true_length[l] << endl;
	}

	// 予測と正解の誤差の平均・分散
	cout << "\e[1mError:\e[0m" << endl;
	cout << "\e[1mL	Mean		StdDev\e[0m" << endl;
	for(int l = 0;l < max_word_length;l++){
		double mean = compute_mean(errors[l]);
		cout << l + 1 << ":	" << fixed << setprecision(5) << mean << ":	" << compute_stddev(errors[l], mean) << endl;
	}

	delete glm;
}
