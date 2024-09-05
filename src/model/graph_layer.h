#pragma once

#include "layer.h"

namespace cpp_mlp {
class GraphLayer : public Layer {
 public:
  explicit GraphLayer(size_t);
  GraphLayer(const size_t, Layer*);
  ~GraphLayer(){};

 private:
  std::vector<Neuron> neurons_;

  auto GetWeightByIndex(size_t) -> const weight& override;
  auto ValueToArray() -> std::vector<double> override;
  void SetNeuronsValue(const std::vector<double>) override;
  void SetWeightByIndex(size_t, weight&) override;

  void CalcErrorOutput(const size_t) override;
  void CalcValueNeurons() override;
  void CalcErrorHidden() override;
  void UpdateWeight() override;
  void RandomFill() override;
  void Print() override;

  auto GetNeurons() -> std::vector<Neuron>& { return neurons_; };
  auto CalcErrorRate(int) -> double;
};
}  // namespace cpp_mlp