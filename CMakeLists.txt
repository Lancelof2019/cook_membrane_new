##
#  CMake script for the step-<XYZ> tutorial program:
##

# Set the name of the project and target:
SET(TARGET "cook_membrane")
#PROJECT(${TARGET} C CXX)
# Declare all source files the target consists of:
SET(TARGET_SRC
  ${TARGET}.cc
  gcmma/GCMMASolver.cpp
  mma/MMASolver.cpp
  # You can specify additional files here!
  )


#add_executable(cook_membrane cook_membrane.cc)

SET(CLEAN_UP_FILES
  # a custom list of globs, e.g. *.log *.vtk
  *.vtk
)

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)

include_directories(/usr/include/trilinos)
include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include)
include_directories(/usr/include/suitesparse)
include_directories(/usr/lib/petscdir/petsc3.15/x86_64-linux-gnu-real/include)

#add_library(mma_gcmma OBJECT gcmma/GCMMASolver.cpp)
#add_library(mma_mma OBJECT mma/MMASolver.cpp)





#find_package(OpenMP REQUIRED)
#target_link_libraries(cook_membrane PUBLIC OpenMP::OpenMP_CXX)



FIND_PACKAGE(deal.II 9.3.0 QUIET 
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()
#target_link_libraries(cook_membrane mma_gcmma mma_mma ${DEAL_II_LIBRARIES})

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()


find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenMP REQUIRED)

add_library(mma_gcmma OBJECT gcmma/GCMMASolver.cpp)
add_library(mma_mma OBJECT mma/MMASolver.cpp)

SET(CLEAN_UP_FILES *.log *.gmv *.gnuplot *.gpl *.eps *.pov *.ucd *.d2 *.vtu *.pvtu)
MESSAGE(STATUS "deal.II_DIR: ${deal.II_DIR}")
MESSAGE(STATUS "DEAL_II_DIR: ${DEAL_II_DIR}")
target_link_libraries(${TARGET} Eigen3::Eigen OpenMP::OpenMP_CXX)
