#include "matrix.h"

namespace cpp_mlp {
Matrix::Matrix() { rows_ = cols_ = 0; }

Matrix::Matrix(size_t rows, size_t cols) {
  if (rows < 1 || cols < 1) {
    throw std::invalid_argument("Wrong matrix size");
  }

  rows_ = rows;
  cols_ = cols;

  buffer_.resize(rows, std::vector<double>(cols));
}

Matrix::Matrix(const Matrix& other) : Matrix(other.rows_, other.cols_) {
  *this = other;
}

void Matrix::RandomFill() {
  static std::mt19937 dice(time(nullptr));
  std::for_each(buffer_.begin(), buffer_.end(), [](auto&& iv) {
    for (auto&& i : iv) {
      i = [](double fMin, double fMax) -> double {
        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(fMin, fMax);
        return dist(e2);
      }(-0.5, 0.5);
    }
  });
}

void Matrix::Clear() {
  if (!buffer_.empty()) {
    for (auto& row : buffer_) row.clear();
    buffer_.clear();
  }
  rows_ = 0;
  cols_ = 0;
}

void Matrix::Print() {
  std::for_each(buffer_.begin(), buffer_.end(), [](const auto& iv) {
    for (const auto& i : iv) {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  });
  std::cout << std::endl;
}

double& Matrix::operator()(size_t rows, size_t cols) {
  if (rows >= rows_ || cols >= cols_) {
    throw std::out_of_range("Array index out of bound");
  }
  return buffer_[rows][cols];
}

double Matrix::operator()(size_t rows, size_t cols) const {
  if (rows >= rows_ || cols >= cols_) {
    throw std::out_of_range("Array index out of bound");
  }
  return buffer_[rows][cols];
}

Matrix Matrix::Transpose() {
  Matrix resultMatrix(cols_, rows_);

  for (size_t i = 0; i < rows_; i++) {
    for (size_t j = 0; j < cols_; j++) {
      resultMatrix.buffer_[j][i] = buffer_[i][j];
    }
  }
  return resultMatrix;
}

void Matrix::MulNumber(const double num) {
  for (size_t i = 0; i < rows_; i++) {
    for (size_t j = 0; j < cols_; j++) {
      buffer_[i][j] *= num;
    }
  }
}

void Matrix::SubMatrix(const Matrix& other) {
  Matrix resultMatrix(rows_, cols_);
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::invalid_argument("Different dimensions of matrices");
  }

  for (size_t i = 0; i < rows_; i++) {
    for (size_t j = 0; j < cols_; j++) {
      buffer_[i][j] -= other.buffer_[i][j];
    }
  }
}

void Matrix::MulMatrix(const Matrix& other) {
  if (cols_ != other.rows_) {
    throw std::invalid_argument(
        "The columns count first matrix is not equal rows of the "
        "second matrix");
  }

  Matrix tempMatrix(rows_, other.cols_);
  for (size_t i = 0; i < rows_; i++) {
    for (size_t j = 0; j < other.cols_; j++) {
      for (size_t k = 0; k < cols_; k++) {
        tempMatrix.buffer_[i][j] += buffer_[i][k] * other.buffer_[k][j];
      }
    }
  }
  *this = tempMatrix;
}

Matrix& Matrix::operator=(const Matrix& other) {
  if (this == &other) {
    return *this;
  }

  rows_ = other.rows_;
  cols_ = other.cols_;
  buffer_ = other.GetBuffer();

  return *this;
}

Matrix Matrix::operator-(const Matrix& other) {
  Matrix resultMatrix(*this);
  resultMatrix.SubMatrix(other);
  return resultMatrix;
}

Matrix Matrix::operator+(const Matrix& other) {
  if (cols_ != other.cols_ || other.rows_ != rows_) {
    throw std::invalid_argument(
        "+ The columns/rows count first matrix is not equal columns/rows of "
        "the "
        "second matrix");
  }

  Matrix resultMatrix(*this);

  for (size_t i = 0; i < rows_; i++) {
    for (size_t j = 0; j < other.cols_; j++) {
      resultMatrix(i, j) += other(i, j);
    }
  }
  return resultMatrix;
}

Matrix Matrix::operator*(const Matrix& other) {
  Matrix resultMatrix(*this);
  resultMatrix.MulMatrix(other);
  return resultMatrix;
}

Matrix Matrix::operator*(const double num) {
  Matrix resultMatrix(*this);
  resultMatrix.MulNumber(num);
  return resultMatrix;
}

Matrix operator*(double num, const Matrix& other) {
  Matrix resultMatrix(other);
  return resultMatrix * num;
}

void Matrix::operator-=(const Matrix& other) { SubMatrix(other); }

}  // namespace cpp_mlp
