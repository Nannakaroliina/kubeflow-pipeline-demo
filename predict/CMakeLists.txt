cmake_minimum_required(VERSION 3.9.2)
project(predict)

add_executable(predict predict.cpp)

find_package(EDDL REQUIRED)
target_link_libraries(predict PUBLIC EDDL::eddl)