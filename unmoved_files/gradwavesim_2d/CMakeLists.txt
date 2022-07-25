add_library(gradwavesim_vem2
    src/sim.cpp
    include/vem/gradwavesim_2d/sim.hpp
    src/sim_viewer.cpp
    include/vem/gradwavesim_2d/sim_viewer.hpp
    )
TARGET_LINK_LIBRARIES(gradwavesim_vem2 PUBLIC poisson_vem2 vem mtao::visualization)
target_include_directories(gradwavesim_vem2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)


