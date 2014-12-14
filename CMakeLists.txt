cmake_minimum_required (VERSION 2.8)
project(hdr)
add_definitions(-Wall -O2 -std=c++11 -g -Wno-unused-local-typedefs)
find_package(JPEG REQUIRED)

include_directories(~/eigen include)
#add_subdirectory(src) # genere lib for project
include_directories(include)
file(GLOB SRC_FILES src/*.cpp include/*.hpp include/*.h)
add_library(libProjectSrc ${SRC_FILES})

add_executable(hdr  main.cpp)
target_link_libraries(hdr libProjectSrc jpeg )

