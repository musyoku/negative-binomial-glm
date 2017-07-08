#include "../model.cpp"
using namespace std;

int main(int argc, char *argv[]){
	PyTrainer* trainer = new PyTrainer(8, 1, 4, 0.2);
	trainer->add_textfile("../../../dataset/japanese.txt");
	trainer->compile();
	cout << "#words: " << trainer->get_num_words() << endl;
	cout << "#characters: " << trainer->get_num_characters() << endl;

	int itr = 1;
	while(true){
		trainer->perform_mcmc();
		if(itr % 1000 == 0){
			cout << "itr: " << itr << " - log likelihood: " << trainer->compute_joint_log_likelihood() << " - MCMC acceptance: " << trainer->get_acceptance_rate() << " - precision: " << trainer->compute_mean_precision(0.99, 20) << endl;
		}
		if(itr % 10000 == 0){
			trainer->save("glm.model");
		}
		itr++;
	}
	delete trainer;
}
