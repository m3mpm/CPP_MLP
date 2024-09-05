#ifndef SRC_MLP_CONTROLLER_H_
#define SRC_MLP_CONTROLLER_H_

#include "../model/network_service.h"
#include "../model/neural_network.h"

namespace cpp_mlp {
class Controller {
 public:
  Controller() : nn_{nullptr}, ns_{std::make_shared<NetworkService>()} {}

  ~Controller() {}

  void InitNeuralNetwork(bool, size_t);

  void SetWeights(const std::string&);
  void SetDataset(const std::string&);
  void SetStatisticFunc(std::function<void(Metrics)>);
  void SetProgressFunc(std::function<void(int)>);

  size_t GetResult(const std::vector<double>&);

  void SaveWeight(const std::string&);
  void Train(size_t);
  void Experiment(double = 1.);
  void CrossValidation(size_t k);

  void StopWork();

 private:
  std::shared_ptr<NetworkService> ns_;
  cpp_mlp::NetworkService::neural_ptr nn_;
};
}  // namespace cpp_mlp

#endif  // SRC_MLP_CONTROLLER_H_
