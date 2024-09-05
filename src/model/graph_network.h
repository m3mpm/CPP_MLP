#pragma once

#include "neural_network.h"
#include "timer.h"

namespace cpp_mlp {
class GraphNetwork : public NeuralNetwork {
 public:
  explicit GraphNetwork(size_t count_hidden) : NeuralNetwork(count_hidden) {}
  ~GraphNetwork() = default;

 private:
  Layer* GetInstance(size_t size) override { return new GraphLayer(size); }
  Layer* GetInstance(size_t size, Layer* prev) override {
    return new GraphLayer(size, prev);
  }
};
}  // namespace cpp_mlp