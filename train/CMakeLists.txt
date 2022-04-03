cmake_minimum_required(VERSION 3.9.2)
project(train)

add_executable(train train.cpp)

find_package(EDDL REQUIRED)
target_link_libraries(train PUBLIC EDDL::eddl)
