# CMAKE FOR REGRESSION MODELS
#

# Add core executables
pybind11_add_module(models models.cpp)										# SET THE NAME OF THE MODULE

# Link the pybind11 library.
target_link_libraries(models PRIVATE pybind11::pybind11)								# SET THE NAME OF THE MODULE

# Link the python311 library.
link_directories(C:/Users/caleb/AppData/Local/Programs/Python/Python311/libs)

# Include Python Development Eviroment.
set(Python3_USE_DEBUG_EXTENSION FALSE)
find_package(Python3 3.11 COMPONENTS Development REQUIRED)
target_link_libraries(models PRIVATE Python3::Python)									# SET THE NAME OF THE MODULE