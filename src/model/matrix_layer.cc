#include "matrix_layer.h"

namespace cpp_mlp {
MatrixLayer::MatrixLayer(size_t size)
    : neurons_(std::make_shared<ColVector>(size)) {
  prev_ = nullptr;
  next_ = nullptr;
  size_ = size;
}

MatrixLayer::MatrixLayer(size_t size, Layer* prev) : MatrixLayer(size) {
  prev_ = prev;
  errors_ = std::make_shared<ColVector>(size_);
  weights_ = std::make_shared<Matrix>(size_, prev_->GetSize());
  delta_weights_ = std::make_shared<Matrix>(size_, prev_->GetSize());

  prev_->SetNextLayer(this);
};

void MatrixLayer::Activate() {
  std::for_each(neurons_->begin(), neurons_->end(),
                [](std::vector<double>& x) { ActivationFunction(x[0]); });
}

void MatrixLayer::CalcValueNeurons() {
  *neurons_ = *GetWeights() * *dynamic_cast<MatrixLayer*>(prev_)->GetNeurons();
  Activate();
}

void MatrixLayer::CalcErrorHidden() {
  ColVector error_rate(
      std::move(dynamic_cast<MatrixLayer*>(next_)->GetWeights()->Transpose() *
                dynamic_cast<MatrixLayer*>(next_)->GetErrors()));

  *errors_ = neurons_->MulElement(1 - *neurons_).MulElement(error_rate);
}

void MatrixLayer::UpdateWeight() {
  delta_weights_->MulNumber(kLearningRate);

  *delta_weights_ =
      *delta_weights_ +
      kLearningRate * (1 - kLearningRate) * *errors_ *
          dynamic_cast<MatrixLayer*>(prev_)->GetNeurons()->Transpose();

  *weights_ -= *delta_weights_;
}

void MatrixLayer::CalcErrorOutput(const size_t key) {
  for (size_t i = 0; i < size_; ++i) {
    SetErrorByIndex(i, -(*neurons_)[i] * (1. - (*neurons_)[i]) *
                           ((i == key - 1) - (*neurons_)[i]));
  }
}

std::vector<double> MatrixLayer::ValueToArray() {
  size_t size = neurons_->size();
  std::vector<double> result(size);
  for (size_t i = 0; i < size; ++i) {
    result[i] = (*neurons_)[i];
  }
  return result;
}

void MatrixLayer::SetNeuronsValue(const std::vector<double> value) {
  *neurons_ = value;
}

const weight& MatrixLayer::GetWeightByIndex(size_t i) {
  return weights_->GetBufferRef()[i];
}

void MatrixLayer::SetWeightByIndex(size_t i, weight& weight) {
  weights_->GetBufferRef()[i] = weight;
}

void MatrixLayer::RandomFill() { weights_->RandomFill(); }

void MatrixLayer::ActivationFunction(double& x) {
  x = (1. / (1. + std::exp(-x)));
}

void MatrixLayer::SetErrorByIndex(size_t i, double error) {
  (*errors_)[i] = error;
}

void MatrixLayer::Print() {
  std::cout << "              neurons             " << std::endl;
  neurons_->Print();

  if (weights_) {
    std::cout << "              weights             " << std::endl;
    weights_->Print();

    std::cout << "              errors_             " << std::endl;
    errors_->Print();
  }
  std::cout << "----------------------------" << std::endl;
}

}  // namespace cpp_mlp
