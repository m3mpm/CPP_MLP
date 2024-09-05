#include "neural_network.h"

#include "timer.h"

namespace cpp_mlp {
NeuralNetwork::NeuralNetwork(size_t size) : progress{nullptr} { size_ = size; }

NeuralNetwork::~NeuralNetwork() {
  for (auto& layer : layers_) delete layer;
};

void NeuralNetwork::SetDataset(const std::shared_ptr<Dataset>& data) {
  dataset_ = data;
}

size_t NeuralNetwork::GetSize() { return size_; }

void NeuralNetwork::InitLayers() {
  layers_.clear();
  layers_.emplace_back(GetInstance(kInputNeurons));
  for (size_t i = 0; i < size_; ++i) {
    layers_.emplace_back(GetInstance(kHiddenNeurons, layers_.back()));
  }
  layers_.emplace_back(GetInstance(kOutputNeurons, layers_.back()));
}

void NeuralNetwork::Train(size_t from, size_t to) {
  Timer timer;
  size_t one_percent = (to - from) / 100;

  const auto& dataset = dataset_->GetData();
  for (size_t i = from; i < to && work_; ++i) {
    PropagateForward(*(dataset[i]));
    PropagateBackward();

    if (progress != nullptr) {
      if (i % one_percent == 0) {
        progress((i - from) / one_percent + 1);
      }
    }
  }
}

ConfusionMatrix NeuralNetwork::Experiment(size_t from, size_t to) {
  auto& dataset = dataset_->GetData();

  ConfusionMatrix confusion{0, 0, 0, 0, 0};
  size_t one_percent = (to - from) / 100;

  for (size_t i = from; i < to && work_; ++i) {
    PropagateForward(*(dataset[i]));
    CalcConfusion(dataset[i]->GetKey(), &confusion);

    if (progress != nullptr) {
      if (i % one_percent == 0) {
        progress((i - from) / one_percent + 1);
      }
    }
  }
  return confusion;
}

int NeuralNetwork::Experiment(const Picture pic) {
  PropagateForward(pic);
  auto actual_values = layers_.back()->ValueToArray();
  return std::distance(
      actual_values.begin(),
      std::max_element(actual_values.begin(), actual_values.end()));
}

void NeuralNetwork::PropagateForward(const Picture& pic) const {
  layers_.front()->SetNeuronsValue(pic.GetData());

  std::for_each(std::next(layers_.begin()), layers_.end(),
                [](Layer* x) { x->CalcValueNeurons(); });
  layers_.back()->CalcErrorOutput(pic.GetKey());
}

void NeuralNetwork::PropagateBackward() {
  std::for_each(std::next(layers_.rbegin()), std::prev(layers_.rend()),
                [](Layer* layer) { layer->CalcErrorHidden(); });

  std::for_each(layers_.rbegin(), std::prev(layers_.rend()),
                [](Layer* layer) { layer->UpdateWeight(); });
}

void NeuralNetwork::CalcConfusion(int key, ConfusionMatrix* confusions) {
  auto actual_values = layers_.back()->ValueToArray();

  int result = std::distance(
      actual_values.begin(),
      std::max_element(actual_values.begin(), actual_values.end()));

  std::vector<double> expected_values(kOutputNeurons);
  expected_values[key - 1] = 1.0;

  for (size_t i = 0; i < kOutputNeurons; ++i) {
    double delta_value = expected_values[i] - actual_values[i];

    confusions->true_negative +=
        (delta_value < 0) && (actual_values[i] < kBorderTruth);
    confusions->true_positive +=
        (delta_value > 0) && (actual_values[i] > kBorderTruth);
    confusions->false_positive +=
        (delta_value < 0) && (actual_values[i] > kBorderTruth);
    confusions->false_negative +=
        (delta_value > 0) && (actual_values[i] < kBorderTruth);
  }

  confusions->result = result;
}

void NeuralNetwork::SetWeight(std::vector<weight>&& weights) {
  size_t counter = 0;
  std::for_each(std::next(layers_.begin()), layers_.end(), [&](auto* layer) {
    for (size_t i = 0, size = layer->GetSize(); i < size; ++i) {
      layer->SetWeightByIndex(i, weights[counter++]);
    }
  });
}

void NeuralNetwork::GetAllWeights(std::vector<weight>& all_weight) {
  all_weight.resize(kHiddenNeurons * size_ + kOutputNeurons);
  size_t counter = 0;
  std::for_each(std::next(layers_.begin()), layers_.end(), [&](auto* layer) {
    for (size_t i = 0, size = layer->GetSize(); i < size; ++i) {
      all_weight[counter++] = layer->GetWeightByIndex(i);
    }
  });
}

void NeuralNetwork::RandomFillWeights() {
  std::for_each(std::next(layers_.begin()), layers_.end(),
                [](Layer*& layer) { layer->RandomFill(); });
}

void NeuralNetwork::Print() {
  for (auto& layers : layers_) {
    layers->Print();
  }
}

void ConfusionMatrix::Print() {
  std::cout << "TP: " << true_positive << std::endl
            << "TN: " << true_negative << std::endl
            << "FP: " << false_positive << std::endl
            << "FN: " << false_negative << std::endl;
}
}  // namespace cpp_mlp
