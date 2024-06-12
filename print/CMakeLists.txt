# CMAKE FILE FOR PRINT PROJECT 
#

# Add core executables
pybind11_add_module(print print.cpp)

# Find packages
#find_package(fmt REQUIRED)

# Link packages
#target_link_libraries(print fmt::fmt)
target_link_libraries(print PRIVATE pybind11::pybind11)

# Link the python312 library.
link_directories(C:/Users/caleb/AppData/Local/Programs/Python/Python312/libs)

# Include Python Development Eviroment.
find_package(Python3 COMPONENTS Development REQUIRED)
target_link_libraries(print PRIVATE Python3::Python)