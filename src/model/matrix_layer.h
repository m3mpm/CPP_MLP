#pragma once

#include "layer.h"

namespace cpp_mlp {
class MatrixLayer : public Layer {
 public:
  using vector_ptr = std::shared_ptr<ColVector>;
  using matrix_ptr = std::shared_ptr<Matrix>;

  explicit MatrixLayer(size_t);
  MatrixLayer(const size_t, Layer*);
  ~MatrixLayer(){};

 private:
  matrix_ptr delta_weights_;
  matrix_ptr weights_;
  vector_ptr neurons_;
  vector_ptr errors_;

  auto GetWeightByIndex(size_t i) -> const weight& override;
  auto ValueToArray() -> std::vector<double> override;
  void SetNeuronsValue(const std::vector<double>) override;
  void SetWeightByIndex(size_t, weight&) override;

  void CalcErrorOutput(const size_t) override;
  void CalcValueNeurons() override;
  void CalcErrorHidden() override;
  void UpdateWeight() override;
  void RandomFill() override;
  void Print() override;

  auto GetWeights() -> matrix_ptr& { return weights_; }
  auto GetNeurons() -> vector_ptr& { return neurons_; }
  auto GetErrors() -> ColVector& { return *errors_; }
  void SetError(Matrix&& error) { *errors_ = error; }

  static void ActivationFunction(double& x);
  void SetErrorByIndex(size_t, double);
  void Activate();
};
}  // namespace cpp_mlp
