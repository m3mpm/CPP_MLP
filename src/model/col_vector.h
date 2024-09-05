#pragma once

#include "matrix.h"

namespace cpp_mlp {
class ColVector : public Matrix {
 public:
  using iterator = cpp_mlp::Matrix::matrix_t::iterator;

  ColVector() : Matrix(1, 1) {}
  explicit ColVector(const Matrix&);
  explicit ColVector(const Matrix&&);
  explicit ColVector(size_t rows) : Matrix(rows, 1) {}

  auto operator=(const std::vector<double>&) -> ColVector&;
  auto operator=(const Matrix&) -> ColVector&;
  auto operator[](size_t) -> double&;
  auto operator[](size_t) const -> double;

  friend ColVector operator-(double, const ColVector&);
  ColVector MulElement(const ColVector&);

  iterator begin() { return GetBufferRef().begin(); }
  iterator end() { return GetBufferRef().end(); }
  size_t size() { return GetBuffer().size(); }
};
}  // namespace cpp_mlp