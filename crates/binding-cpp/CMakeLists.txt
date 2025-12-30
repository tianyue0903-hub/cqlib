cmake_minimum_required(VERSION 3.12)
project(cqlib_cpp_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

# -----------------------------------------------------------------------------
# 1. Configuration & Auto-detection
# -----------------------------------------------------------------------------

# Default to Debug if CMAKE_BUILD_TYPE is not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Map CMake build type to Cargo profile
if(CMAKE_BUILD_TYPE MATCHES "Release")
    set(CARGO_PROFILE "release")
    set(CARGO_FLAGS "--release")
else()
    set(CARGO_PROFILE "debug")
    set(CARGO_FLAGS "")
endif()

# Determine Rust library name based on platform
if (WIN32)
    set(RUST_LIB_NAME "binding_cpp.lib")
else()
    set(RUST_LIB_NAME "libbinding_cpp.a")
endif()

# Determine Cargo target directory (respect env var if set)
if(DEFINED ENV{CARGO_TARGET_DIR})
    set(RUST_TARGET_DIR "$ENV{CARGO_TARGET_DIR}")
else()
    set(RUST_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../target")
endif()

set(RUST_LIB "${RUST_TARGET_DIR}/${CARGO_PROFILE}/${RUST_LIB_NAME}")

message(STATUS "Configuring for ${CMAKE_BUILD_TYPE} build")
message(STATUS "Rust library path: ${RUST_LIB}")

# -----------------------------------------------------------------------------
# 2. Integrate Cargo Build
# -----------------------------------------------------------------------------

find_program(CARGO_EXECUTABLE cargo REQUIRED)

# Create a custom target to build the Rust library
add_custom_target(cargo_build
    COMMAND ${CARGO_EXECUTABLE} build --package binding-cpp ${CARGO_FLAGS}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    COMMENT "Building Rust library binding-cpp (${CARGO_PROFILE})..."
    VERBATIM
)

# -----------------------------------------------------------------------------
# 3. Define C++ Executable
# -----------------------------------------------------------------------------

add_executable(cpp_test tests/test.cpp)

# -----------------------------------------------------------------------------
# 4. Include Paths
# -----------------------------------------------------------------------------

# Include directory for CXX generated headers
include_directories("${RUST_TARGET_DIR}/cxxbridge")
# Include current source dir to find local headers
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

# -----------------------------------------------------------------------------
# 5. Linking
# -----------------------------------------------------------------------------

# Ensure Rust library is built before linking
add_dependencies(cpp_test cargo_build)

# Link the Rust library
target_link_libraries(cpp_test PRIVATE ${RUST_LIB})

# Link system libraries
if (UNIX)
    target_link_libraries(cpp_test PRIVATE pthread dl)
    if (APPLE)
        target_link_libraries(cpp_test PRIVATE "-framework CoreFoundation")
    endif()
endif()

if (WIN32)
    target_link_libraries(cpp_test PRIVATE ws2_32 userenv advapi32 bcrypt)
endif()
