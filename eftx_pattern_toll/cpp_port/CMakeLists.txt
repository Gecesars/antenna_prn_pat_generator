cmake_minimum_required(VERSION 3.16)

project(EFTXPatternToolCpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(eftx_core
    src/util.cpp
    src/io.cpp
    src/resample.cpp
    src/metrics.cpp
    src/export.cpp
    src/project.cpp
    src/dashboard.cpp
)
target_include_directories(eftx_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(eftx_pattern_tool_cpp
    src/main.cpp
)
target_link_libraries(eftx_pattern_tool_cpp PRIVATE eftx_core)

install(TARGETS eftx_pattern_tool_cpp RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

set(CPACK_PACKAGE_NAME "EFTX Pattern Tool Cpp")
set(CPACK_PACKAGE_VENDOR "EFTX")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "EFTXPatternToolCpp")
set(CPACK_GENERATOR "ZIP")

include(CPack)
