cmake_minimum_required(VERSION 3.12)
project(cqlib_cpp_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

# 1. 找到 Cargo 编译出来的静态库
# 注意：这里假设你已经运行了 cargo build --release -p binding-cpp
# Linux/Mac 是 libbinding_cpp.a，Windows 是 binding_cpp.lib
if (WIN32)
    set(RUST_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release/binding_cpp.lib")
else()
    set(RUST_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release/libbinding_cpp.a")
endif()

# 2. 定义可执行文件
add_executable(cpp_test tests/test.cpp)


# 3. 告诉 C++ 编译器去哪里找头文件
# cxx 生成的头文件通常在这个目录下
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../target/cxxbridge")
# 同时也包含当前源码目录，以便能找到 lib.rs.h
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

# 4. 链接 Rust 库
target_link_libraries(cpp_test PRIVATE ${RUST_LIB})

# 5. (Mac/Linux 必须) 链接系统基础库
if (UNIX)
    # Rust 依赖 pthread 和 dl
    target_link_libraries(cpp_test PRIVATE pthread dl)
    if (APPLE)
        # Mac 可能还需要 CoreFoundation
        target_link_libraries(cpp_test PRIVATE "-framework CoreFoundation")
    endif()
endif()

if (WIN32)
    # Windows 需要链接系统库
    target_link_libraries(cpp_test PRIVATE ws2_32 userenv advapi32 bcrypt)
endif()
