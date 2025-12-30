use cqlib_core::add;

#[cxx::bridge(namespace = "cqlib")]
mod ffi {
    // 暴露给 C++ 调用的 Rust 函数列表
    extern "Rust" {
        // 定义一个简单的静态函数
        fn add(left: u64, right: u64) -> u64;
    }
}

#[test]
fn it_works() {
    let result = add(2, 2);
    assert_eq!(result, 4);
}
