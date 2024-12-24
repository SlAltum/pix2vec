cmake_minimum_required(VERSION 3.10.0)
project(Pix2Vec VERSION 0.1.0 LANGUAGES C CXX)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()

add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(GTest REQUIRED)
find_package(OpenCLHeaders REQUIRED)
find_package(OpenCLICDLoader REQUIRED)
find_package(OpenCLHeadersCpp REQUIRED)

include_directories("include")
include_directories(3rdparty/stb)
file(GLOB SOURCES src/*.cpp)

add_library(Pix2Vec STATIC ${SOURCES})
target_link_libraries(Pix2Vec PRIVATE OpenCL::Headers OpenCL::OpenCL OpenCL::HeadersCpp)
target_compile_definitions(Pix2Vec PRIVATE CL_HPP_TARGET_OPENCL_VERSION=300)

if (WIN32)
    if ("${CMAKE_MAKE_PROGRAM}" MATCHES "MsBuild")
        add_custom_command(TARGET Pix2Vec
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/$(ConfigurationName)/assets
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/kernel ${CMAKE_BINARY_DIR}/$(ConfigurationName)/kernel
                )
    elseif(MINGW)
        add_custom_command(TARGET Pix2Vec
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/kernel ${CMAKE_BINARY_DIR}/kernel
                )
    endif ()
endif ()

install(TARGETS Pix2Vec RUNTIME DESTINATION bin)

add_executable(Pix2VecTest test/pix2vec_test.cpp)
target_link_libraries(Pix2VecTest PRIVATE ${GTEST_BOTH_LIBRARIES} pthread)
target_link_libraries(Pix2VecTest PRIVATE Pix2Vec)

gtest_discover_tests(Pix2VecTest)