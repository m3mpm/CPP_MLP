#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

namespace cpp_mlp {
class Picture {
 public:
  Picture(int key, const std::vector<double>& value)
      : key_(key), value_(value) {}
  explicit Picture(const std::vector<double>& value) : value_(value) {}

  auto GetKey() const -> size_t { return key_; }
  auto GetKey() -> size_t& { return key_; }

  std::vector<double>& GetData() { return value_; }
  std::vector<double> GetData() const { return value_; }

 private:
  size_t key_{1};
  std::vector<double> value_;
};

class Dataset {
 public:
  using iterator = std::vector<Picture*>::iterator;

  ~Dataset() {
    for (auto& data : dataset_) delete data;
  }

  void Shuffle() {
    std::shuffle(dataset_.begin(), dataset_.end(),
                 std::default_random_engine());
  }

  void Print() {
    std::cout << "Key: " << dataset_[0]->GetKey() << std::endl;
    for (auto& i : dataset_[0]->GetData()) {
      std::cout << "Data: " << i << std::endl;
    }
  }

  auto GetData() -> std::vector<Picture*>& { return dataset_; }
  auto begin() -> iterator { return dataset_.begin(); }
  auto end() -> iterator { return dataset_.end(); }
  auto size() -> size_t { return dataset_.size(); }

 private:
  std::vector<Picture*> dataset_;
};
}  // namespace cpp_mlp
