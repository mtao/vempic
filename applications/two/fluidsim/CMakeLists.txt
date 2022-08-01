if(VEM_USE_OPENGL)
    SET(VIEWER_SRCS
    src/sim_viewer.cpp
    include/vem/fluidsim_2d/sim_viewer.hpp
        )
endif()
add_library(fluidsim_vem2
    src/sim.cpp
    #src/sim_serialization.cpp
    #src/sim_advection.cpp
    #src/sim_pressure.cpp
    #src/sim_scene.cpp
    #src/cell.cpp
    src/fluidvem.cpp
    #src/fluidvem2_function_projection.cpp
    #src/sim_buoyancy.cpp
    #${VIEWER_SRCS}
    )
#find_package(CHOLMOD REQUIRED)


target_compile_options(fluidsim_vem2 PUBLIC -DVEM_FLUX_MOMENT_FLUID)
TARGET_LINK_LIBRARIES(fluidsim_vem2 PUBLIC poisson_vem2 vem std::filesystem
    # spqr ${CHOLMOD_LIBRARIES}
    nlohmann_json::nlohmann_json)
if(VEM_USE_PYTHON)
    TARGET_LINK_LIBRARIES(fluidsim_vem2 PUBLIC
    mtao::python)
target_compile_options(fluidsim_vem2 PUBLIC -DVEM_USE_PYTHON)
endif()

#target_include_directories(fluidsim_vem2 PUBLIC ${CHOLMOD_INCLUDES})
if(VEM_USE_OPENGL)
    TARGET_LINK_LIBRARIES(fluidsim_vem2 PUBLIC mtao::visualization
        )
ENDIF()

target_include_directories(fluidsim_vem2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)


