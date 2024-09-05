#pragma once

#include <list>
#include <memory>
#include <vector>

#include "col_vector.h"
#include "neuron.h"

namespace cpp_mlp {
class Layer {
 public:
  virtual ~Layer(){};

  virtual auto GetWeightByIndex(size_t) -> const weight& = 0;
  virtual auto ValueToArray() -> std::vector<double> = 0;
  virtual void SetNeuronsValue(const std::vector<double>) = 0;
  virtual void SetWeightByIndex(size_t, weight&) = 0;

  virtual void CalcErrorOutput(size_t) = 0;
  virtual void CalcValueNeurons() = 0;
  virtual void CalcErrorHidden() = 0;
  virtual void UpdateWeight() = 0;
  virtual void RandomFill() = 0;
  virtual void Print() = 0;

  void SetNextLayer(Layer* next) { next_ = next; }
  size_t GetSize() { return size_; }

 protected:
  Layer* prev_;
  Layer* next_;
  size_t size_;
};
}  // namespace cpp_mlp