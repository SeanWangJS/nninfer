#include <iostream> 

#include <variant>


int main(int argc, char** argv) {

    std::variant<std::pair<int, int>, int> kernel_size = 1;
    std::cout << kernel_size.index() << std::endl;

    // kernel_size = std::make_pair(1, 2);
    // std::cout << kernel_size.index() << std::endl;

    int d = std::get<int>(kernel_size);
    std::cout << d << std::endl;
}