cmake_minimum_required(VERSION 3.10)

project(VLADImageSearch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(vlad_search src/main.cpp)

target_include_directories(vlad_search PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vlad_search PRIVATE ${OpenCV_LIBS})

# ---------------------------------------------------------------
# Geometric verification (Meghana): library + test program
# ---------------------------------------------------------------
add_library(geometric_verifier src/GeometricVerifier.cpp)
target_include_directories(geometric_verifier PUBLIC ${CMAKE_SOURCE_DIR}/src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(geometric_verifier PUBLIC ${OpenCV_LIBS})

add_executable(verify_pair apps/verify_pair.cpp)
target_link_libraries(verify_pair PRIVATE geometric_verifier)
