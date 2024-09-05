#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

namespace cpp_mlp {
class Matrix {
 public:
  using matrix_t = std::vector<std::vector<double> >;

  Matrix();
  explicit Matrix(size_t rows, size_t cols);
  Matrix(const Matrix& other);

  ~Matrix() = default;

  auto GetBuffer() const -> const matrix_t& { return buffer_; }
  auto GetBufferRef() -> matrix_t& { return buffer_; }
  auto GetCols() const -> size_t { return cols_; }
  auto GetRows() const -> size_t { return rows_; }

  void MulMatrix(const Matrix&);
  void SubMatrix(const Matrix&);
  void MulNumber(double);
  auto Transpose() -> Matrix;

  auto operator()(size_t, size_t) -> double&;
  auto operator()(size_t, size_t) const -> double;
  auto operator-(const Matrix&) -> Matrix;
  auto operator+(const Matrix&) -> Matrix;
  auto operator*(const Matrix&) -> Matrix;
  auto operator*(const double) -> Matrix;
  auto operator=(const Matrix&) -> Matrix&;
  void operator-=(const Matrix&);

  friend auto operator*(double, const Matrix&) -> Matrix;
  void RandomFill();
  void Print();
  void Clear();

 private:
  matrix_t buffer_;
  size_t rows_;
  size_t cols_;
};
}  // namespace cpp_mlp
