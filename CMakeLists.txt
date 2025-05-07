cmake_minimum_required(VERSION 3.14)
project(ProiectPI1)

set(CMAKE_CXX_STANDARD 17)

# --- Find OpenCV
set(OpenCV_DIR "D:/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_DIR}/x64/vc16/lib)
set(OpenCV_LIBRARIES opencv_world4110)

add_executable(ProiectPI1
        main.cpp
        Resizing.cpp
        Resizing.h
)

target_link_libraries(ProiectPI1 ${OpenCV_LIBRARIES})

set_target_properties(ProiectPI1 PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ProiectPI1"
)

