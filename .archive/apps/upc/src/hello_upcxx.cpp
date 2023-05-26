#include <upcxx/upcxx.hpp>
#include <iostream>

int main() {
upcxx::init();

std::cout<<"Hello from "<<upcxx::rank_me()<<" of "<<upcxx::rank_n()<<std::endl;

upcxx::finalize();
return 0;
}
