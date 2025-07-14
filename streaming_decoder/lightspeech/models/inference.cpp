#include <torch/script.h>
#include <iostream>
# include <memory>

int main(){
    torch::jit::script::Module module;
    try{
        module = torch::jit::load("/home/naiscorp/Desktop/streaming_asr/streaming_decoder/lightspeech/models/module_prototype.pt");
    }
    catch (const c10::Error& e){
        std::cerr << "Error loading the model" << std::endl;
        std::cerr << e.what() << std::endl;
        return -1;
    }

    std::cout << "Model loaded successfully!" << std::endl;

    torch::Tensor input = torch::randn({10, 1, 5});
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    torch::jit::IValue output = module.forward(inputs);

    auto out_list = output.toList();
    for (size_t i = 0; i < out_list.size(); ++i){
        auto tuple = out_list.get(i).toTuple();
        auto elements = tuple->elements();

        std::cout << "Element " << i << ":\n";
        for (size_t j = 0; j < elements.size(); ++j){
            std::cout << "Tensor " << j << ":\n" << elements[j].toTensor() << "\n";
        }
    }

    return 0;
}