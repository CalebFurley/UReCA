# CMAKE FILE FOR PRINT PROJECT 
#

# Add core executables
add_executable (print "print.h" "print.cpp")

# Find packages
find_package(fmt REQUIRED)
#find_package(pybind11 REQUIRED)

# Link packages
target_link_libraries(print fmt::fmt)
#target_link_libraries(print pybind11::pybind11)

###
### PLAN IS TO MAKE THIS PROJECT INTO A GIT REPO, THEN ADD PYBIND11 AS A SUBMODULE.
###