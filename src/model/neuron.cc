#include "neuron.h"

namespace cpp_mlp {

void Neuron::FillRandomWeight(int size_prev_layer) {
  weights_.resize(size_prev_layer);
  delta_weights_.resize(size_prev_layer);
  std::generate(weights_.begin(), weights_.end(), ChoiceDoubleRand);
}

double Neuron::ChoiceDoubleRand() {
  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<double> random_double(-0.5, 0.5);
  return random_double(eng);
}

void Neuron::CalcValue(const std::vector<double>& neurons_value) {
  int size = weights_.size();
  double value = 0;
  for (int i = 0; i < size; ++i) {
    value += neurons_value[i] * weights_[i];
  }
  value_ = 1. / (1 + std::exp(-value));
}

void Neuron::SetWeight(std::vector<double>& data) {
  weights_.resize(data.size());
  delta_weights_.resize(data.size());
  std::copy(data.begin(), data.end(), weights_.begin());
}

void Neuron::UpdateWeight(const std::vector<double>& signal) {
  for (size_t i = 0, size = weights_.size(); i < size; ++i) {
    double current_delta =
        kLearningRate * delta_weights_[i] +
        kLearningRate * (1 - kLearningRate) * signal[i] * error_;
    delta_weights_[i] = current_delta;
    weights_[i] -= current_delta;
  }
}

void Neuron::CalcError(double error_rate) {
  error_ = value_ * (1 - value_) * error_rate;
}

}  // namespace cpp_mlp
