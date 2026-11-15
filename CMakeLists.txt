cmake_minimum_required(VERSION 3.16)
project(vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "$ENV{HOME}/aplicaciones/Librerias/opencvi" ${CMAKE_PREFIX_PATH})
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio viz)

message(STATUS "OpenCV version:  ${OpenCV_VERSION}")
message(STATUS "OpenCV include:  ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs:     ${OpenCV_LIBS}")



add_executable(vision.bin practica.cpp)
target_include_directories(vision.bin PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision.bin PRIVATE ${OpenCV_LIBS})
