#include "controller.h"

namespace cpp_mlp {
void Controller::InitNeuralNetwork(bool type_neural_network,
                                   size_t num_hidden_layers) {
  if (type_neural_network == 0) {
    nn_ = std::make_shared<MatrixNetwork>(num_hidden_layers);
  } else {
    nn_ = std::make_shared<GraphNetwork>(num_hidden_layers);
  }
  ns_->SetNeuralNetwork(nn_);
}

void Controller::SetWeights(const std::string& weights_path) {
  ns_->LoadWeight(weights_path);
}

void Controller::SetDataset(const std::string& file) { ns_->LoadDataset(file); }

void Controller::SetStatisticFunc(std::function<void(Metrics)> func_ptr) {
  ns_->SetStatisticFunc(func_ptr);
}

void Controller::SetProgressFunc(std::function<void(int)> func_ptr) {
  ns_->SetProgressFunc(func_ptr);
}

void Controller::Train(size_t epoch_amount) { ns_->Train(epoch_amount); }

void Controller::Experiment(double rate) { ns_->Experiment(rate); }

size_t Controller::GetResult(const std::vector<double>& pic_data) {
  return ns_->Experiment(pic_data);
}

void Controller::SaveWeight(const std::string& savepath) {
  ns_->SaveWeight(savepath.c_str());
}

void Controller::CrossValidation(size_t k) { ns_->CrossValidation(k); }

void Controller::StopWork() { ns_->StopWork(); }

}  // namespace cpp_mlp
