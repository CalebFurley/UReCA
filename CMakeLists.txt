# GLOBAL URECA SOLUTION CONFIGURATION
#

# Set the minimum version of CMake required to build the project.
cmake_minimum_required (VERSION 3.8)

# !!! Important !!! Set the toolchain file to the vcpkg.cmake file in your vcpkg directory.
set(CMAKE_TOOLCHAIN_FILE "C:/Users/caleb/VCPKG/vcpkg/scripts/buildsystems/vcpkg.cmake")


# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# Set the solution name.
project ("UReCA")

# Inlcude Eigen module for linear algebra.
include_directories(${CMAKE_SOURCE_DIR}/external/eigen)

# Include sub-projects here..
add_subdirectory (pybind11) # Include the pybind11 library first.
add_subdirectory ("print")
add_subdirectory ("models")
add_subdirectory ("tools") 
