
if(VEM_USE_OPENGL)
    SET(VIEWER_SRCS
    src/sim_viewer.cpp
        )
endif()
add_library(wavesim_vem2
    src/sim.cpp
    ${VIEWER_SRCS}
    )
TARGET_LINK_LIBRARIES(wavesim_vem2 PUBLIC poisson_vem2 vem)
if(VEM_USE_OPENGL)
    TARGET_LINK_LIBRARIES(wavesim_vem2 PUBLIC mtao::visualization

mtao::python
        )
ADD_EXECUTABLE(wavesim_gui_example2
    src/wavesim_gui_example2.cpp
    )
TARGET_LINK_LIBRARIES(wavesim_gui_example2 wavesim_vem2 vem_visualize
    mtao::visualization MagnumPlugins::PngImageConverter)
ENDIF()
target_include_directories(wavesim_vem2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)


