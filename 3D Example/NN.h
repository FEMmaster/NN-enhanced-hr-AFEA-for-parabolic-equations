#pragma once

#include <torch/torch.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/schedulers/lr_scheduler.h>
#include <torch/data/dataloader.h>
#include <torch/data/samplers/random.h>
#include <torch/data/transforms/stack.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <random>


class SimpleDataset : public torch::data::datasets::Dataset<SimpleDataset> {
public:
    SimpleDataset(torch::Tensor data, torch::Tensor labels)
        : data_(data), labels_(labels) {}

    torch::data::Example<> get(size_t index) override {
        return { data_[index], labels_[index] };
    }

    torch::optional<size_t> size() const override {
        return data_.size(0);
    }

    void shuffle() {
        auto indices = torch::randperm(data_.size(0));
        data_ = data_.index_select(0, indices);
        labels_ = labels_.index_select(0, indices);
    }

    torch::Tensor& data() { return data_; }
    torch::Tensor& labels() { return labels_; }

private:
    torch::Tensor data_, labels_;
};


namespace torch {
    namespace optim {

        class CustomScheduler : public LRScheduler {
        public:
            CustomScheduler(torch::optim::Optimizer& optimizer, double initial_lr, double alpha)
                : LRScheduler(optimizer), optimizer_(optimizer), initial_lr_(initial_lr), alpha_(alpha) {}

            std::vector<double> get_lrs() override {
                std::vector<double> new_lrs;
                unsigned epoch = step_count_;  // 使用基类中的step_count_跟踪epoch
                for (auto& param_group : optimizer_.param_groups()) {
                    double new_lr = initial_lr_ / (1 + alpha_ * epoch);
                    new_lrs.push_back(new_lr);
                }
                return new_lrs;
            }

        protected:
            torch::optim::Optimizer& optimizer_;
            double initial_lr_;
            double alpha_;
        };

    } // namespace optim
} // namespace torch

struct Net : torch::nn::Module {
    std::vector<torch::nn::Linear> layers;
    std::unique_ptr<torch::optim::Optimizer> optimizer;

    Net(std::vector<int> layers_config) {

        // 使用 std::random_device 生成随机种子
        std::random_device rd;
        auto random_seed = rd();

        // 设置 PyTorch 的随机种子
        torch::manual_seed(random_seed);

        for (size_t i = 0; i < layers_config.size() - 1; ++i) {
            auto linear_layer = register_module("fc" + std::to_string(i),
                torch::nn::Linear(layers_config[i], layers_config[i + 1]));

            // 初始化权重和偏置
            torch::nn::init::xavier_uniform_(linear_layer->weight);     // for tanh
            //torch::nn::init::kaiming_uniform_(linear_layer->weight);    // for relu
            torch::nn::init::constant_(linear_layer->bias, 0.0);

            layers.push_back(linear_layer);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            x = torch::tanh(layers[i]->forward(x));                     // for xavier_uniform_
            //x = torch::relu(layers[i]->forward(x));                     // for kaiming_uniform_
        }
        x = layers.back()->forward(x);
        return x;
    }

    void train(torch::Tensor inputs, torch::Tensor targets, size_t epochs, double initial_lr) {
        optimizer = std::make_unique<torch::optim::Adam>(this->parameters(), torch::optim::AdamOptions(initial_lr));
        //optimizer = std::make_unique<torch::optim::SGD>(this->parameters(), torch::optim::SGDOptions(initial_lr));

        double alpha = 0.0005;  // 学习率衰减因子
        //double alpha = 0;  // 学习率衰减因子
        torch::optim::CustomScheduler scheduler(*optimizer, initial_lr, alpha);  // Pass both initial_lr and alpha

        for (size_t epoch = 1; epoch <= epochs; ++epoch) {
            optimizer->zero_grad();
            auto prediction = this->forward(inputs);
            auto loss = torch::mse_loss(prediction, targets);
            loss.backward();
            optimizer->step();
            scheduler.step();  // 更新学习率

            if (epoch % 100 == 0) {
                std::cout << "Epoch [" << epoch << "] loss: " << loss.item<float>() << std::endl;
            }

            // Check if the loss is less than 5e-5, and if so, break out of the loop.
            if (loss.item<float>() < 1.0e-4) {
                std::cout << "Early stopping triggered at epoch " << epoch << " with loss: " << loss.item<float>() << std::endl;
                break;
            }
        }

        optimizer.reset();  // 释放Adam优化器

        // 创建LBFGS优化器并进行设置
        torch::optim::LBFGSOptions lbfgs_options(1e-2);
        lbfgs_options.max_iter(50000);
        lbfgs_options.max_eval(50000);
        lbfgs_options.tolerance_grad(1e-5);
        lbfgs_options.tolerance_change(1e-9);
        //lbfgs_options.history_size(50);
        lbfgs_options.line_search_fn("strong_wolfe");

        auto lbfgs_optimizer = std::make_unique<torch::optim::LBFGS>(this->parameters(), lbfgs_options);

        // 使用LBFGS优化器进行进一步优化，使用closure函数
        size_t lbfgs_step = 0;
        auto closure = [this, &inputs, &targets, &lbfgs_step]() {
            this->zero_grad();
            auto prediction = this->forward(inputs);
            auto loss = torch::mse_loss(prediction, targets);
            loss.backward();
            lbfgs_step++;
            if (lbfgs_step % 100 == 0) {
                std::cout << "LBFGS Step [" << lbfgs_step << "] loss: " << loss.item<float>() << std::endl;
            }
            return loss;
            };
        lbfgs_optimizer->step(closure);
    }

    void train_minibatch(torch::Tensor inputs, torch::Tensor targets, size_t epochs, double initial_lr, size_t batch_size) {
        // 创建数据集实例
        auto dataset = SimpleDataset(inputs, targets).map(torch::data::transforms::Stack<>());

        // 创建 DataLoader
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            dataset, torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

        // 初始化优化器
        optimizer = std::make_unique<torch::optim::Adam>(this->parameters(), torch::optim::AdamOptions(initial_lr));

        double alpha = 0.0005;  // 学习率衰减因子
        torch::optim::CustomScheduler scheduler(*optimizer, initial_lr, alpha);

        for (size_t epoch = 1; epoch <= epochs; ++epoch) {
            float epoch_loss = 0.0;
            size_t batch_count = 0;

            for (auto& batch : *data_loader) {
                auto batch_data = batch.data;
                auto batch_labels = batch.target;

                optimizer->zero_grad();
                auto prediction = this->forward(batch_data);
                auto loss = torch::mse_loss(prediction, batch_labels);
                loss.backward();
                optimizer->step();

                epoch_loss += loss.item<float>();
                batch_count++;
                //std::cout << "batch_count: " << batch_count << " loss: " << loss.item<float>() << std::endl;
            }

            scheduler.step();  // 更新学习率每个epoch之后

            if (epoch % 100 == 0) {  // 只在每100次epoch后打印
                epoch_loss /= batch_count;
                std::cout << "Epoch [" << epoch << "] Average loss: " << epoch_loss << std::endl;
            }

            // Early stopping condition based on the epoch average loss
            if (epoch_loss < 1.0e-4) {
                std::cout << "Early stopping triggered at epoch " << epoch << " with average loss: " << epoch_loss << std::endl;
                break;
            }
        }
        optimizer.reset();  // 释放Adam优化器

        // 创建LBFGS优化器并进行设置
        torch::optim::LBFGSOptions lbfgs_options(1e-2);
        lbfgs_options.max_iter(50000);
        lbfgs_options.max_eval(50000);
        lbfgs_options.tolerance_grad(1e-5);
        lbfgs_options.tolerance_change(1e-9);
        lbfgs_options.history_size(50);
        lbfgs_options.line_search_fn("strong_wolfe");

        auto lbfgs_optimizer = std::make_unique<torch::optim::LBFGS>(this->parameters(), lbfgs_options);

        // 使用LBFGS优化器进行进一步优化，使用closure函数
        size_t lbfgs_step = 0;
        auto closure = [this, &inputs, &targets, &lbfgs_step]() {
            this->zero_grad();
            auto prediction = this->forward(inputs);
            auto loss = torch::mse_loss(prediction, targets);
            loss.backward();
            lbfgs_step++;
            if (lbfgs_step % 1 == 0) {
                std::cout << "LBFGS Step [" << lbfgs_step << "] loss: " << loss.item<float>() << std::endl;
            }
            return loss;
            };
        lbfgs_optimizer->step(closure);
    }
};