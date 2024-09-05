#include "gtest/gtest.h"
#include "model/network_service.h"
#include "model/neural_network.h"

const constexpr size_t kInputNeurons = 784;
size_t num_hidden_layers = 2;
std::string dataset_file = "resources/datasets/emnist-letters-train.csv";
std::string test_file = "resources/datasets/emnist-letters-test-one.csv";
std::string weights_file = "resources/weigths/weigths_2_layers.csv";

std::shared_ptr<cpp_mlp::NetworkService> ns_ =
    std::make_shared<cpp_mlp::NetworkService>();
cpp_mlp::NetworkService::neural_ptr nn_;

void setCurrentValues(cpp_mlp::Metrics) {}
std::function<void(cpp_mlp::Metrics)> statistic_ptr = &setCurrentValues;

TEST(graph, test_1) {
  nn_ = std::make_shared<cpp_mlp::GraphNetwork>(num_hidden_layers);
  ns_->SetNeuralNetwork(nn_);
  ns_->LoadDataset(dataset_file);
  ns_->SetStatisticFunc(statistic_ptr);
  ns_->Train(1);
  ns_->LoadDataset(test_file);
  ASSERT_EQ(21, (nn_->Experiment(0, 1).result + 1));
}

TEST(graph, test_2) {
  nn_ = std::make_shared<cpp_mlp::GraphNetwork>(num_hidden_layers);
  ns_->SetNeuralNetwork(nn_);
  ns_->SetStatisticFunc(statistic_ptr);
  ns_->LoadWeight(weights_file);
  ns_->LoadDataset(test_file);
  ASSERT_EQ(21, (nn_->Experiment(0, 1).result + 1));
}

TEST(graph, test_3) {
  nn_ = std::make_shared<cpp_mlp::GraphNetwork>(num_hidden_layers);
  ns_->SetNeuralNetwork(nn_);
  ns_->LoadDataset(dataset_file);
  ns_->SetStatisticFunc(statistic_ptr);
  ns_->CrossValidation(5);
  ns_->LoadDataset(test_file);
  ASSERT_EQ(21, (nn_->Experiment(0, 1).result + 1));
}

TEST(matrix, test_1) {
  nn_ = std::make_shared<cpp_mlp::MatrixNetwork>(num_hidden_layers);
  ns_->SetNeuralNetwork(nn_);
  ns_->LoadDataset(dataset_file);
  ns_->SetStatisticFunc(statistic_ptr);
  ns_->Train(1);
  ns_->LoadDataset(test_file);
  ASSERT_EQ(21, (nn_->Experiment(0, 1).result + 1));
}

TEST(matrix, test_2) {
  nn_ = std::make_shared<cpp_mlp::MatrixNetwork>(num_hidden_layers);
  ns_->SetNeuralNetwork(nn_);
  ns_->SetStatisticFunc(statistic_ptr);
  ns_->LoadWeight(weights_file);
  ns_->LoadDataset(test_file);
  ASSERT_EQ(21, (nn_->Experiment(0, 1).result + 1));
}

TEST(matrix, test_3) {
  nn_ = std::make_shared<cpp_mlp::MatrixNetwork>(num_hidden_layers);
  ns_->SetNeuralNetwork(nn_);
  ns_->LoadDataset(dataset_file);
  ns_->SetStatisticFunc(statistic_ptr);
  ns_->CrossValidation(5);
  ns_->LoadDataset(test_file);
  ASSERT_EQ(21, (nn_->Experiment(0, 1).result + 1));
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
