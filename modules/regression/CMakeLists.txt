## CMake for Regression Module.
#

# Add core executables
pybind11_add_module(regression regression.cpp)  # SET THE NAME OF THE MODULE

# Link the pybind11 library.
target_link_libraries(regression PRIVATE pybind11::module)  # SET THE NAME OF THE MODULE

# Include Python Development Environment.
set(Python3_USE_DEBUG_EXTENSION FALSE)
find_package(Python3 3.12 COMPONENTS Development REQUIRED)
target_link_libraries(regression PRIVATE Python3::Python)  # SET THE NAME OF THE MODULE

# Include the directory containing the header files (just in case.)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
