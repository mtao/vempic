
if(VEM_USE_OPENGL)
    SET(VIEWER_SRCS
    src/sim_viewer.cpp
    include/vem/wavesim_2d/sim_viewer.hpp
        )
endif()
add_library(wavesim_vem2
    src/sim.cpp
    include/vem/wavesim_2d/sim.hpp
    ${VIEWER_SRCS}
    )
TARGET_LINK_LIBRARIES(wavesim_vem2 PUBLIC poisson_vem2 vem)
if(VEM_USE_OPENGL)
    TARGET_LINK_LIBRARIES(wavesim_vem2 PUBLIC mtao::visualization

mtao::python
        )
ENDIF()
target_include_directories(wavesim_vem2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)


