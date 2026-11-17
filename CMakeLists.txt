cmake_minimum_required(VERSION 3.10)

project(VLADImageSearch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(vlad_search src/main.cpp)

target_include_directories(vlad_search PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vlad_search PRIVATE ${OpenCV_LIBS})
