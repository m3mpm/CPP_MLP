#include "network_service.h"

namespace cpp_mlp {

NetworkService::NetworkService(neural_ptr neural_network)
    : neural_network_(neural_network){};

void NetworkService::LoadDataset(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) throw std::invalid_argument("Ошибка при открытии файла");

  dataset_ = std::make_shared<cpp_mlp::Dataset>();
  auto& dataset_value = dataset_->GetData();

  while (!file.eof()) {
    std::string line, cell;
    std::getline(file, line);

    if (line.empty()) continue;

    std::stringstream lineStream(line);
    std::getline(lineStream, cell, ',');
    int mapping_code = stod(cell);

    std::vector<double> tmp(kInputNeurons);

    for (int i = 0; getline(lineStream, cell, ','); ++i) {
      tmp[i] = (stod(cell) / 255.0 * 0.99 + 0.01);
    }
    dataset_value.emplace_back(new Picture(mapping_code, tmp));
  }
  neural_network_->SetDataset(dataset_);
}

void NetworkService::SaveWeight(const std::string& filename) {
  std::vector<weight> all_weight;
  neural_network_->GetAllWeights(all_weight);
  std::ofstream out(filename);
  if (out.is_open()) {
    for (auto& weights : all_weight) {
      for (auto weight : weights) {
        out << weight << ',';
      }
      out << std::endl;
    }
  }
  out.close();
}

void NetworkService::LoadWeight(const std::string& filename) {
  std::ifstream in(filename);
  if (in.is_open()) {
    std::string line;
    std::vector<weight> weights_vec(
        kHiddenNeurons * neural_network_->GetSize() + kOutputNeurons);
    for (int i = 0; getline(in, line); ++i) {
      std::stringstream lineStream(line);
      double weight{0};
      while (getline(lineStream >> weight, line, ',')) {
        weights_vec[i].push_back(weight);
      }
    }
    neural_network_->SetWeight(std::move(weights_vec));
  }
  in.close();
}

void NetworkService::SetNeuralNetwork(neural_ptr neural_network) {
  neural_network_ = neural_network;
  neural_network_->SetDataset(dataset_);
  neural_network_->InitLayers();
}

void NetworkService::SetDatasetInNetwork() {
  neural_network_->SetDataset(dataset_);
}

void NetworkService::Train(size_t epoch) {
  neural_network_->SetStopOrWork(true);

  Timer timer;
  neural_network_->RandomFillWeights();
  dataset_->Shuffle();
  size_t start_train_pos = dataset_->size() / 10;

  for (size_t i = 0; i < epoch; ++i) {
    neural_network_->Train(start_train_pos, dataset_->size());
    CalcMetrix(neural_network_->Experiment(0, start_train_pos),
               timer.PastTime());
  }
}

void NetworkService::Experiment(double rate) {
  neural_network_->SetStopOrWork(true);

  Timer timer;
  auto statistic =
      neural_network_->Experiment(0, dataset_->size() * std::min(1., rate));
  CalcMetrix(statistic, timer.PastTime());
}

int NetworkService::Experiment(const std::vector<double>& pic) {
  return neural_network_->Experiment(Picture{pic});
}

void NetworkService::CrossValidation(size_t k) {
  Timer timer;
  neural_network_->SetStopOrWork(true);
  size_t step = dataset_->size() / k;
  neural_network_->RandomFillWeights();

  for (size_t i = 0, datasize = dataset_->size(); i < k; ++i) {
    size_t testset_from = i * step;
    size_t testset_to = (i == k - 1) ? datasize : (i + 1) * step;

    neural_network_->Train(0, testset_from);
    neural_network_->Train(testset_to, datasize);

    auto statistic = neural_network_->Experiment(testset_from, testset_to);

    CalcMetrix(statistic, timer.PastTime());
  }
}

void NetworkService::CalcMetrix(ConfusionMatrix conf, double total_time) {
  metrics_.accuracy =
      (conf.true_positive + conf.true_negative) /
      static_cast<double>(conf.true_positive + conf.true_negative +
                          conf.false_negative + conf.false_positive);

  metrics_.precision =
      conf.true_positive /
      static_cast<double>(conf.true_positive + conf.false_positive);

  metrics_.recall =
      conf.true_positive /
      static_cast<double>(conf.true_positive + conf.false_negative);

  metrics_.f_measure =
      (2 * metrics_.precision * metrics_.recall) /
      static_cast<double>(metrics_.precision + metrics_.recall);

  metrics_.total_time = total_time;

  if (neural_network_->IsWorked()) statistic_func_ptr(metrics_);
}

void NetworkService::PrintMetrix() {
  std::cout << "Accuracy: " << metrics_.accuracy << std::endl
            << "Precision: " << metrics_.precision << std::endl
            << "Recall: " << metrics_.recall << std::endl
            << "F-measure: " << metrics_.f_measure << std::endl;
}

}  // namespace cpp_mlp
