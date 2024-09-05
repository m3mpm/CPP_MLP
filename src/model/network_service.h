#pragma once

#include <functional>
#include <memory>

#include "graph_network.h"
#include "matrix_network.h"
#include "metrics.h"

namespace cpp_mlp {
class NetworkService {
 public:
  using neural_ptr = std::shared_ptr<NeuralNetwork>;

  NetworkService(neural_ptr = nullptr);
  ~NetworkService() {}

  void SetNeuralNetwork(neural_ptr);
  void LoadDataset(const std::string&);
  void LoadWeight(const std::string&);
  void SaveWeight(const std::string&);
  void SetDatasetInNetwork();

  void CrossValidation(size_t);
  void Experiment(double = 1.);
  void Train(size_t);

  void PrintMetrix();
  void Print() { neural_network_->Print(); }

  void SetProgressFunc(std::function<void(int)> func_ptr) {
    neural_network_->SetProgressFunc(func_ptr);
  }

  void SetStatisticFunc(std::function<void(Metrics)> func_ptr) {
    statistic_func_ptr = func_ptr;
  }

  int Experiment(const std::vector<double>& pic);

  void StopWork() { neural_network_->SetStopOrWork(false); }

 private:
  neural_ptr neural_network_;
  std::shared_ptr<Dataset> dataset_;
  Metrics metrics_;

  std::function<void(Metrics)> statistic_func_ptr;

  void CalcMetrix(ConfusionMatrix, double);
};
}  // namespace cpp_mlp
