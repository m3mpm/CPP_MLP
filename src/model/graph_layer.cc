#include "graph_layer.h"

namespace cpp_mlp {
GraphLayer::GraphLayer(size_t size) {
  prev_ = nullptr;
  next_ = nullptr;
  size_ = size;
  neurons_.resize(size);
}

GraphLayer::GraphLayer(const size_t size, Layer* prev) : GraphLayer(size) {
  prev_ = prev;
  prev->SetNextLayer(this);
}

void GraphLayer::RandomFill() {
  std::for_each(neurons_.begin(), neurons_.end(),
                [this](Neuron& x) { x.FillRandomWeight(prev_->GetSize()); });
}

void GraphLayer::SetNeuronsValue(const std::vector<double> input_data) {
  for (size_t i = 0; i < neurons_.size(); ++i) {
    neurons_[i].SetValue(input_data[i]);
  }
}

void GraphLayer::CalcValueNeurons() {
  auto values_array = prev_->ValueToArray();
  for (size_t i = 0; i < size_; ++i) {
    neurons_[i].CalcValue(values_array);
  }
}

void GraphLayer::CalcErrorOutput(const size_t key) {
  auto actual_values = ValueToArray();
  for (size_t i = 0; i < size_; ++i) {
    neurons_[i].SetError(-actual_values[i] * (1. - actual_values[i]) *
                         ((i == key - 1) - actual_values[i]));
  }
}

double GraphLayer::CalcErrorRate(int index) {
  double error_rate = 0;
  for (size_t i = 0; i < size_; ++i) {
    error_rate += neurons_[i].GetWeightByIndex(index) * neurons_[i].GetError();
  }
  return error_rate;
}

void GraphLayer::CalcErrorHidden() {
  for (size_t i = 0; i < size_; ++i) {
    double error_rate = dynamic_cast<GraphLayer*>(next_)->CalcErrorRate(i);
    neurons_[i].CalcError(error_rate);
  }
}

void GraphLayer::UpdateWeight() {
  auto neurons_value = prev_->ValueToArray();
  for (size_t i = 0; i < size_; ++i) {
    neurons_[i].UpdateWeight(neurons_value);
  }
}

std::vector<double> GraphLayer::ValueToArray() {
  size_t size = neurons_.size();
  std::vector<double> result(size);
  for (size_t i = 0; i < size; ++i) {
    result[i] = neurons_[i].GetValue();
  }
  return result;
}

const weight& GraphLayer::GetWeightByIndex(size_t i) {
  return neurons_[i].GetWeight();
}

void GraphLayer::SetWeightByIndex(size_t i, weight& weight) {
  neurons_[i].SetWeight(weight);
}

void GraphLayer::Print() {
  for (size_t i = 0; i < size_; ++i) {
    printf("%.4f\t{ ", neurons_[i].GetValue());
    for (size_t j = 0; j < neurons_[i].GetWeight().size(); ++j) {
      printf("%+.2f ", neurons_[i].GetWeightByIndex(j));
    }
    printf("}\t[ %+.2f ]\n", neurons_[i].GetError());
  }
  std::cout << "-------------------------" << std::endl;
}

}  // namespace cpp_mlp
