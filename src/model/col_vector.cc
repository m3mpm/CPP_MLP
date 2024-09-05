#include "col_vector.h"

namespace cpp_mlp {
ColVector::ColVector(const Matrix& other) : Matrix(other.GetRows(), 1) {
  *this = other;
}

ColVector::ColVector(const Matrix&& other) : Matrix(other.GetRows(), 1) {
  *this = std::move(other);
}

ColVector& ColVector::operator=(const std::vector<double>& value) {
  ColVector result(value.size());
  for (size_t i = 0, size = value.size(); i < size; ++i) {
    result[i] = value[i];
  }
  *this = result;
  return *this;
}

ColVector& ColVector::operator=(const Matrix& other) {
  Matrix::operator=(other);
  return *this;
}

ColVector ColVector::MulElement(const ColVector& other) {
  if (GetRows() != other.GetRows()) {
    throw std::invalid_argument(
        "Ошибка ColVector operator*(const ColVector& other)");
  }

  size_t size = GetRows();
  ColVector result(size);
  for (size_t i = 0; i < size; ++i) {
    result[i] = (*this)[i] * other[i];
  }
  return result;
}

double& ColVector::operator[](size_t index) { return GetBufferRef()[index][0]; }

double ColVector::operator[](size_t index) const {
  return GetBuffer()[index][0];
}

ColVector operator-(double num, const ColVector& vec) {
  size_t size = vec.GetRows();
  ColVector result(size);
  for (size_t i = 0; i < size; ++i) {
    result[i] = num - vec[i];
  };
  return result;
}

}  // namespace cpp_mlp
