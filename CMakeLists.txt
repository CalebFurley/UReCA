﻿# CMake file for UReCA project.
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# Set the solution name.
project ("UReCA")

# Include Eigen module for linear algebra.
include_directories(${CMAKE_SOURCE_DIR}/libraries/eigen)

# Include pybind11 module
add_subdirectory(${CMAKE_SOURCE_DIR}/libraries/pybind11)

# Find Python
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development)

# Include sub-projects here..
add_subdirectory ("modules/classification")
add_subdirectory ("modules/regression")
add_subdirectory ("modules/toolset")
