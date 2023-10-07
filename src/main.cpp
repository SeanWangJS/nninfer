#include <iostream> 
#include <fstream>
#include <vector>
#include <string>


int main(int argc, char** argv) {

    std::string path = "./data/data.npy";

    std::ifstream infile(path, std::ios::binary);
    
    char header[118];
    infile.read(header, 118);

    std::cout << header << std::endl;

    std::vector<int> shape;
    int ndim;
    infile.read(reinterpret_cast<char*>(&ndim), sizeof(int));
    std::cout << "ndim: " << ndim << std::endl;


}