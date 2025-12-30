#include <iostream>

#include "binding-cpp/src/lib.rs.h"

int main() {
    std::cout << "C++: rust add ..." << std::endl;

    // cqlib::add 来自我们在 Rust bridge 里定义的 namespace="cqlib"
    auto res = cqlib::add(2, 8);

    std::cout << "Result: " << res << std::endl;
    return 0;
}
