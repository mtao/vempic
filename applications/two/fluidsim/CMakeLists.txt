if(VEM_USE_OPENGL)
    SET(VIEWER_SRCS
    src/sim_viewer.cpp
        )
endif()
add_library(fluidsim_vem2
    src/sim.cpp
    src/sim_serialization.cpp
    src/sim_advection.cpp
    src/sim_pressure.cpp
    src/sim_scene.cpp
    #src/cell.cpp
    src/fluidvem.cpp
    #src/fluidvem_function_projection.cpp
    src/sim_buoyancy.cpp
    ${VIEWER_SRCS}
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


add_executable(fluidsim_2d src/fluidsim_2d.cpp)
target_link_libraries(fluidsim_2d vem fmt fluidsim_vem2)


ADD_EXECUTABLE(fluidsim2_projection_check
    src/fluidsim2_pressure_projection_check.cpp
    )
TARGET_LINK_LIBRARIES(fluidsim2_projection_check fluidsim_vem2 vem)
if(VEM_USE_OPENGL)
ADD_EXECUTABLE(fluidsim_gui_example2
    src/fluidsim_gui_example2.cpp
    )
TARGET_LINK_LIBRARIES(fluidsim_gui_example2 fluidsim_vem2 vem_visualize
    MagnumPlugins::PngImageConverter
    nlohmann_json::nlohmann_json
    )
endif()
