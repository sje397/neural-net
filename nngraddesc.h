#ifndef NNGRADDESC_H
#define NNGRADDESC_H

#include <eigen3/Eigen/Dense>

#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/operations.hpp>

#include <ctime>

namespace sje {

namespace e = Eigen;
namespace fs = boost::filesystem;

typedef double default_real;

template<int nIn, int nHidden, int nOut, int nHiddenLayers = 1, typename real = default_real>
class BackPropagationNeuralNet {
	public:
		typedef e::Matrix<real, e::Dynamic, 1> vec;
		typedef e::Matrix<real, e::Dynamic, e::Dynamic> mat;

		// structure to store neuron states during calculation
		struct State {
			vec valueIn;
			vec valueHidden[nHiddenLayers];
			vec out;

			State() {
				valueIn = vec(nIn + 1);
				valueIn(0) = static_cast<real>(1);
				//#pragma omp parallel for
				for(int i = 0; i < nHiddenLayers; ++i) {
					valueHidden[i] = vec(nHidden + 1);
					valueHidden[i](0) = static_cast<real>(1);
				}
			}
		};

		BackPropagationNeuralNet()
		{
			weightIn = mat::Random(nHidden, nIn + 1);
			weightOut = mat::Random(nOut, nHidden + 1);

			for(int i = 0; i < nHiddenLayers - 1; ++i) {
				weightHidden[i] = mat::Random(nHidden, nHidden + 1);
			}
		}

		virtual ~BackPropagationNeuralNet() {}

		void setWeightFile(const fs::path &path) {
			weightFile = path;
			if(fs::exists(weightFile)) {
				fs::ifstream in(weightFile);

				std::cout << "Loading weights..." << std::endl;

				real val;
				for(int i = 0; i < nHidden; ++i) {
					for(int j = 0; j < nIn + 1; j++) {
						in >> val;
						weightIn(i, j) = val;
					}
				}

				for(int l = 0; l < nHiddenLayers - 1; ++l) {
					for(int i = 0; i < nHidden; ++i) {
						for(int j = 0; j < nHidden + 1; j++) {
							in >> val;
							weightHidden[l](i, j) = val;
						}
					}
				}

				for(int i = 0; i < nOut; ++i) {
					for(int j = 0; j < nHidden + 1; j++) {
						in >> val;
						weightOut(i, j) = val;
					}
				}

			}
		}

		// calculate output for given input and current weights
		vec calc(const vec &in) {
			State state;
			return calc(in, state);
		}

		void train(const vec &in, const vec &ans) {
			lessons.push_back(std::make_pair(in, ans));
		}

		void desc(real alpha = 0.1, real lambda = 1e-6, real threshold = 1e-8) {
			outputs.resize(lessons.size());

			real delta, c = cost(lambda), ctemp, run = c;
			const size_t runlen = 10;

			clock_t start, current, last_save = clock();
			size_t counter = 0;
			do {
				counter++;

				start = clock();

				learn(alpha, lambda);

				ctemp = cost(lambda);
				run = run * (runlen - 1) / runlen + ctemp / runlen;
				delta = ctemp - c;
				c = ctemp;

				current = clock();
				double t = (current - start) / (double)CLOCKS_PER_SEC;
				std::cout << "(" << counter << ") " << "cost: " << c << ", delta: " << delta << ", diff: " << (c - run) << ", time: " << t << std::endl;

				// save weights
				if(current - last_save > 60 * CLOCKS_PER_SEC) {
					last_save = current;
					start = clock();
					if(!weightFile.empty()) {
						fs::path tempFile(weightFile.string() + ".tmp");
						fs::ofstream out(tempFile);
						out << weightIn << std::endl;
						for(int i = 0; i < nHiddenLayers - 1; ++i) {
							out << weightHidden[i] << std::endl;
						}
						out << weightOut << std::endl;
						fs::remove(weightFile);
						fs::rename(tempFile, weightFile);
					}
					current = clock();
					t = (current - start) / (double)CLOCKS_PER_SEC;
					std::cout << "Save time: " << t << std::endl;
				}
			} while(c * std::fabs(c - run) > threshold);
		}

	protected:
		// apply sigmoid function element-wise to a vector
		static vec sigmoid(const vec &in) {
			return ((in.array() * -1).exp() + 1).pow(-1);
		}

		// determine cost for current weights and lessons
		real cost(const real lambda) {
			const vec ones = vec::Ones(nOut);
			real c = 0;

			#pragma omp parallel for reduction(+:c)
			for(int l = 0; l < (int)lessons.size(); ++l) {
				State state;
				calc(lessons.at(l).first, state);
				outputs[l] = state;

				const vec &ans = lessons.at(l).second;

				c = c + (ans.array() * state.out.array().log() + (ones - ans).array() * (ones - state.out).array().log()).sum();
			}

			real ss = weightIn.rightCols(nIn).array().pow(2).sum()
					+ weightOut.rightCols(nOut).array().pow(2).sum();

			#pragma omp parallel for reduction(+:ss)
			for(int i = 0; i < nHiddenLayers - 1; ++i) {
				ss = ss + weightHidden[i].rightCols(nHidden).array().pow(2).sum();
			}

			return (-c + lambda * ss / 2) / lessons.size();
		}

		vec calc(const vec &in, State &state) {
			state.valueIn.tail(nIn) = in;
			state.valueHidden[0].tail(nHidden) = sigmoid(weightIn * state.valueIn);
			for(int i = 1; i < nHiddenLayers; ++i) {
				state.valueHidden[i].tail(nHidden) = sigmoid(weightHidden[i - 1] * state.valueHidden[i - 1]);
			}
			state.out = sigmoid(weightOut * state.valueHidden[nHiddenLayers - 1]);
			return state.out;
		}

		// alpha is learning rate, lambda is regularisation parameter (minimise weight values)
		void learn(const real alpha, const real lambda) {
			// zero error accumulators
			mat errorSumIn = mat::Zero(nHidden, nIn + 1);
			mat errorSumOut = mat::Zero(nOut, nHidden + 1);
			mat errorSumHidden[nHiddenLayers - 1];

			//#pragma omp parallel for
			for(int i = 0; i < nHiddenLayers - 1; ++i) {
				errorSumHidden[i] = mat::Zero(nHidden, nHidden + 1);
			}
			const vec onesh = vec::Ones(nHidden);

			#pragma omp parallel for
			// iterate our training examples and accumulate errors
			for(int l = 0; l < (int)lessons.size(); ++l) {
				const vec &ans = lessons.at(l).second;
				// reuse calcs from cost function instead of:
				//State state;
				//calc(lessons.at(l).first, state);
				const State state = outputs[l];

				const vec errorOut = state.out - ans;
				vec errorHidden[nHiddenLayers];
				errorHidden[nHiddenLayers - 1] = (weightOut.rightCols(nHidden).transpose() * errorOut).array() * state.valueHidden[nHiddenLayers - 1].tail(nHidden).array() * (onesh - state.valueHidden[nHiddenLayers - 1].tail(nHidden)).array();

				for(int i = nHiddenLayers - 2; i >= 0; --i) {
					errorHidden[i] = (weightHidden[i].rightCols(nHidden).transpose() * errorHidden[i + 1]).array() * state.valueHidden[i].tail(nHidden).array() * (onesh - state.valueHidden[i].tail(nHidden)).array();
				}

				#pragma omp critical
				{
					errorSumOut += errorOut * state.valueHidden[nHiddenLayers - 1].transpose();
					for(int i = nHiddenLayers - 2; i >= 0; --i) {
						errorSumHidden[i] += errorHidden[i + 1] * state.valueHidden[i].transpose();
					}
					errorSumIn += errorHidden[0] * state.valueIn.transpose();
				}
			}

			// add error for weight magnitude (regularization) - ignore bias weights
			mat cw1(weightIn);
			cw1.col(0) = vec::Zero(nHidden);
			errorSumIn = errorSumIn / lessons.size() + lambda * cw1;
			mat cw2(weightOut);
			cw2.col(0) = vec::Zero(nOut);
			errorSumOut = errorSumOut / lessons.size() + lambda * cw2;
			//#pragma omp parallel for
			for(int i = 0; i < nHiddenLayers - 1; ++i) {
				mat cwh(weightHidden[i]);
				cwh.col(0) = vec::Zero(nHidden);
				errorSumHidden[i] = errorSumHidden[i] / lessons.size() + lambda * cwh;
			}

			// weight update
			weightIn -= alpha * errorSumIn;
			//#pragma omp parallel for
			for(int i = 0; i < nHiddenLayers - 1; ++i) {
				weightHidden[i] -= alpha * errorSumHidden[i];
			}
			weightOut -= alpha * errorSumOut;
		}

		// weights
		mat weightIn, weightOut;
		mat weightHidden[nHiddenLayers - 1];

		// storage of training inputs
		typedef std::pair<vec, vec> Lesson;
		std::vector<Lesson> lessons;

		// temp storage of outputs - optimisation
		std::vector<State> outputs;

		// persist weights
		fs::path weightFile;
};

}

#endif // NNGRADDESC_H
