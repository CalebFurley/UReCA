# CMAKE FILE FOR PRINT PROJECT 
#

# Add core executables
add_executable (print "print.h" "print.cpp")

# Find packages
#find_package(fmt REQUIRED)

# Link packages
#target_link_libraries(print fmt::fmt)
target_link_libraries(print PRIVATE pybind11::pybind11)