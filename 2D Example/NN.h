#pragma once

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <memory>

struct Net : torch::nn::Module {
    std::vector<torch::nn::Linear> layers;
    std::unique_ptr<torch::optim::Optimizer> optimizer;

    Net(std::vector<int> layers_config) {
        for (size_t i = 0; i < layers_config.size() - 1; ++i) {
            layers.push_back(register_module("fc" + std::to_string(i), torch::nn::Linear(layers_config[i], layers_config[i + 1])));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            x = torch::tanh(layers[i]->forward(x));
            //x = torch::relu(layers[i]->forward(x));
        }
        x = layers.back()->forward(x);
        return x;
    }

    void train(torch::Tensor inputs, torch::Tensor targets, size_t epochs, double learning_rate) {
        optimizer = std::make_unique<torch::optim::Adam>(this->parameters(), torch::optim::AdamOptions(learning_rate));

        for (size_t epoch = 1; epoch <= epochs; ++epoch) {
            optimizer->zero_grad();
            auto prediction = this->forward(inputs);
            auto loss = torch::mse_loss(prediction, targets);
            loss.backward();
            optimizer->step();

            if (epoch % 100 == 0)
                std::cout << "Epoch [" << epoch << "] loss: " << loss.item<float>() << std::endl;

            // Check if the loss is less than 5e-5, and if so, break out of the loop.
            if (loss.item<float>() < 5e-5) {
                std::cout << "Early stopping triggered at epoch " << epoch << " with loss: " << loss.item<float>() << std::endl;
                break;
            }
        }
    }
};




//struct CustomDataset : torch::data::datasets::Dataset<CustomDataset> {
//    torch::Tensor inputs, targets;
//
//    CustomDataset(torch::Tensor inputs, torch::Tensor targets)
//        : inputs(inputs), targets(targets) {}
//
//    torch::data::Example<> get(size_t index) override {
//        return { inputs[index], targets[index] };
//    }
//
//    torch::optional<size_t> size() const override {
//        return inputs.size(0);
//    }
//};
//
//struct Net : torch::nn::Module {
//    torch::nn::Sequential layers;
//
//    Net(const std::vector<int>& layers_config) {
//        // Build the sequential model by looping through the layers_config
//        for (size_t i = 0; i < layers_config.size() - 1; ++i) {
//            layers->push_back(torch::nn::Linear(layers_config[i], layers_config[i + 1]));
//            if (i < layers_config.size() - 2) {
//                // Add ReLU activation for all but the last layer
//                layers->push_back(torch::nn::Functional(torch::relu));
//            }
//        }
//        register_module("layers", layers);
//    }
//
//    torch::Tensor forward(torch::Tensor x) {
//        return layers->forward(x);
//    }
//
//    void train(torch::Tensor inputs, torch::Tensor targets, size_t epochs, double learning_rate) {
//
//        auto dataset = CustomDataset(inputs, targets).map(torch::data::transforms::Stack<>());
//        auto data_loader = torch::data::make_data_loader(
//            dataset,
//            torch::data::DataLoaderOptions().batch_size(1163).workers(4)
//        );
//
//        auto optimizer = std::make_unique<torch::optim::Adam>(this->parameters(), torch::optim::AdamOptions(learning_rate));
//
//        for (int epoch = 1; epoch <= epochs; ++epoch) {
//            for (auto& batch : *data_loader) {
//                auto& data = batch.data;
//                auto& target = batch.target;
//                auto prediction = this->forward(data);
//                auto loss = torch::mse_loss(prediction, target);
//                optimizer->zero_grad();
//                loss.backward();
//                optimizer->step();
//
//                std::cout << "Epoch [" << epoch << "] average loss: " << loss.item<float>() << std::endl;
//            }
//        }
//    }
//};



