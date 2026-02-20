if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.16)
    project(LossLibrary LANGUAGES CXX)
endif()

add_library(LossLibrary INTERFACE)
target_include_directories(LossLibrary INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(LossLibrary INTERFACE MatrixLibrary ToolsLibrary)
