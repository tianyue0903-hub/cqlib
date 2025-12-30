fn main() {
    cxx_build::bridge("src/lib.rs")
        //         .file("src/cqlib_cpp.cc")
        .flag_if_supported("-std=c++11")
        .compile("cqlib_cpp");
}
