#pragma once

#include "data.h"
#include "graph_layer.h"
#include "matrix_layer.h"
#include <functional>

namespace cpp_mlp {
const constexpr size_t kInputNeurons = 784;
const constexpr size_t kHiddenNeurons = 144;
const constexpr size_t kOutputNeurons = 26;
const constexpr double kBorderTruth = 0.7;

struct ConfusionMatrix {
  int true_positive{0};
  int true_negative{0};
  int false_positive{0};
  int false_negative{0};
  int result{0};

  void Print();
};

class NeuralNetwork {
 public:
  NeuralNetwork(size_t);
  ~NeuralNetwork();

  void InitLayers();
  void Train(size_t, size_t);
  void SetWeight(std::vector<weight>&&);
  void GetAllWeights(std::vector<weight>&);
  void SetDataset(const std::shared_ptr<Dataset>&);
  auto Experiment(size_t, size_t) -> ConfusionMatrix;
  void RandomFillWeights();

  auto GetSize() -> size_t;
  auto IsWorked() -> bool { return work_; }
  void Print();

  void SetProgressFunc(std::function<void(int)> func) { progress = func; }

  void SetStopOrWork(bool work) { work_ = work; }

  auto Experiment(const Picture pic) -> int;

 protected:
  std::shared_ptr<Dataset> dataset_;
  std::list<Layer*> layers_;
  size_t size_;
  bool work_ = true;
  std::function<void(int)> progress;

 private:
  virtual Layer* GetInstance(size_t, Layer*) = 0;
  virtual Layer* GetInstance(size_t) = 0;

  void CalcConfusion(int, ConfusionMatrix*);
  void PropagateForward(const Picture&) const;
  void PropagateBackward();
};
}  // namespace cpp_mlp
