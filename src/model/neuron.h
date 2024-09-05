#pragma once

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

namespace cpp_mlp {
const constexpr double kLearningRate = 0.09;

using weight = std::vector<double>;

class Neuron {
 public:
  Neuron() : value_(0), error_(0) {}
  ~Neuron() {}

  void CalcValue(const std::vector<double>&);
  void CalcError(double);

  auto GetWeightByIndex(int i) -> double& { return weights_[i]; }
  auto GetWeight() -> std::vector<double>& { return weights_; }
  auto GetValue() -> double { return value_; }
  auto GetError() -> double { return error_; }

  void SetValue(const double value) { value_ = value; }
  void SetError(const double error) { error_ = error; }
  void SetWeight(std::vector<double>&);

  void UpdateWeight(const std::vector<double>&);
  void FillRandomWeight(int);

 private:
  weight delta_weights_;
  weight weights_;

  double value_;
  double error_;

  static double ChoiceDoubleRand();
};
}  // namespace cpp_mlp
