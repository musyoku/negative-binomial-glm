#include <chrono>
#include "../model.cpp"
using namespace std;

int main(int argc, char *argv[]){
	PyTrainer* trainer = new PyTrainer(8, 1, 4, 0.1);
	trainer->add_textfile("../../../dataset/japanese.txt");
	trainer->compile();
	cout << "#words: " << trainer->get_num_words() << endl;
	cout << "#characters: " << trainer->get_num_characters() << endl;

	int itr = 1;
    auto start = chrono::system_clock::now();
	while(true){
		trainer->perform_mcmc();
		if(itr % 1000 == 0){
		    auto diff = chrono::system_clock::now() - start;
			cout << "itr: " << itr << " - log likelihood: " << trainer->compute_joint_log_likelihood() << " - MCMC acceptance: " << trainer->get_acceptance_rate() << " - precision: " << trainer->compute_mean_precision(0.99, 20) << " - time (s): " << chrono::duration_cast<std::chrono::seconds>(diff).count()
 << endl;
		    start = chrono::system_clock::now();
		}
		if(itr % 10000 == 0){
			trainer->save("glm2.model");
		}
		itr++;
	}
	delete trainer;
}
