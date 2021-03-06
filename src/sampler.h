#pragma once
#include <random>
#include <chrono>

namespace npycrf{
	namespace sampler{
		int seed = std::chrono::system_clock::now().time_since_epoch().count();
		// int seed = 1;
		std::mt19937 mt(seed);
		double gamma(double a, double b){
			std::gamma_distribution<double> distribution(a, 1.0 / b);
			return distribution(mt);
		}
		double beta(double a, double b){
			double ga = gamma(a, 1.0);
			double gb = gamma(b, 1.0);
			return ga / (ga + gb);
		}
		double bernoulli(double p){
			std::uniform_real_distribution<double> rand(0, 1);
			double r = rand(mt);
			if(r > p){
				return 0;
			}
			return 1;
		}
		double uniform(double min = 0, double max = 0){
			std::uniform_real_distribution<double> rand(min, max);
			return rand(mt);
		}
		int randint(int min = 0, int max = 0){
			std::uniform_int_distribution<> rand(min, max - 1);
			return rand(mt);
		}
		double normal(double mean, double stddev){
			std::normal_distribution<double> rand(mean, stddev);
			return rand(mt);
		}
	} // namespace sampler
} // namespace npycrf