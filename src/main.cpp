#include <iostream> 

class A{
    public:
        int* data;
        int d = 10;
        A(int* data){
            this->data = data;
        }
        int* get_data(){
            return data;
        }

        int* get_data() const{
            return data;
        }

        void printFirst() {
            std::cout << data[0] << std::endl;
        }

};

void f(A a) {
    a.printFirst();
    std::cout << a.d << std::endl;
    a.d = 1000;
    a.get_data()[0] = 100;
}

void f2(A &a) {
    a.printFirst();
    std::cout << a.d << std::endl;
    a.d = 1000;
    a.get_data()[0] = 100;
}

int main(int argc, char** argv) {

    int* data = new int[2];
    data[0] = 1;
    data[1] = 2;

    A a(data);
    // f(a); 
    f2(a);
    a.printFirst();
    std::cout << a.d << std::endl;

    return 0;
}