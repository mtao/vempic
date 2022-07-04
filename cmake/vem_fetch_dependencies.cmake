include(FetchContent REQUIRED)


set(MTAO_COMMIT 
    cd56b4a6f897bb5892557179adceaf704fd3ffc0
    )
set(MANDOLINE_COMMIT 
    8955b949ccb302028d370caaf3ea6aa4a95f392b
    )
set(NLOHMANN_JSON_COMMIT
 v3.9.1
    )
set(PROTOBUF_COMMIT v3.11.3)
set(LIBIGL_COMMIT v2.1.0)
set(CATCH_COMMIT v2.9.1)
set(EMBREE_VERSION master)

function(fetch_dep REPO_NAME GIT_REPO GIT_TAG ADD_SUBDIR)
    FetchContent_Declare(
        ${REPO_NAME}
        GIT_REPOSITORY ${GIT_REPO}
        #GIT_TAG f6b406427400ed7ddb56cfc2577b6af571827c8c
        GIT_TAG ${GIT_TAG}
        )
    if(ADD_SUBDIR)
        if(${CMAKE_VERSION} VERSION_LESS 3.14)
            FetchContent_Populate(${REPO_NAME})
            add_subdirectory(${${REPO_NAME}_SOURCE_DIR} ${${REPO_NAME}_BINARY_DIR})
        else()
            FetchContent_MakeAvailable(${REPO_NAME})
        endif()
    else()
        FetchContent_Populate(${REPO_NAME})
    endif()
    set(${REPO_NAME}_SOURCE_DIR ${${REPO_NAME}_SOURCE_DIR} PARENT_SCOPE)
    set(${REPO_NAME}_BINARY_DIR ${${REPO_NAME}_BINARY_DIR} PARENT_SCOPE)
endfunction()



find_package(embree QUIET)
if(NOT embree_FOUND)
    OPTION(EMBREE_TUTORIALS "Enable embree tutorial" OFF)
    fetch_dep(embree https://github.com/mtao/embree.git ${EMBREE_VERSION} ON)
endif()


OPTION(MTAO_USE_ELTOPO "Should we build the el topo submodule" OFF)
OPTION(MTAO_USE_LOSTOPOS "Should we build the LOS Topos submodule" OFF)
OPTION(MTAO_USE_OPENGL "Build opengl stuff" ${VEM_USE_OPENGL})
OPTION(MTAO_USE_PNGPP "Use PNG++ for screenshots" ON)
OPTION(MTAO_USE_PYTHON "Use python " ${VEM_USE_PYTHON})

if(MTAO_PATH)
    ADD_SUBDIRECTORY("${MTAO_PATH}" ${CMAKE_BINARY_DIR}/mtao EXCLUDE_FROM_ALL)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${MTAO_PATH}/cmake")
else()

    fetch_dep(mtao https://github.com/mtao/core.git 
        ${MTAO_COMMIT}
        ON)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${mtao_SOURCE_DIR}/cmake")
endif()





OPTION(MANDOLINE_USE_OPENGL "Build opengl stuff" ${VEM_USE_OPENGL})
if(MANDOLINE_PATH)
    ADD_SUBDIRECTORY("${MANDOLINE_PATH}" ${CMAKE_BINARY_DIR}/mandoline EXCLUDE_FROM_ALL)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${MANDOLINE_PATH}/cmake")
else()

    fetch_dep(mandoline https://github.com/mtao/mandoline.git 
        ${MANDOLINE_COMMIT}
        ON)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${mandoline_SOURCE_DIR}/cmake")
endif()
if(BUILD_TESTING)
    if(NOT Catch2_FOUND)
        fetch_dep(
            catch2
            https://github.com/catchorg/Catch2.git
            v2.9.1
            ON
            )

    endif()
endif()




IF(USE_OPENMP)
    FIND_PACKAGE(OpenMP REQUIRED)
ENDIF(USE_OPENMP)

MESSAGE(STATUS "MODULE PATH:${CMAKE_MODULE_PATH}")
#FIND_PACKAGE(libigl REQUIRED)


if(USE_OPENGL)
    find_package(ImGui COMPONENTS Sources SourcesMiscCpp REQUIRED)
    #find_package(MagnumIntegration COMPONENTS ImGui)
endif()




if(VEM_BUILD_TESTING)
    if(NOT Catch2_FOUND)
        if(NOT TARGET Catch2::Catch2)
        fetch_dep(
            catch2
            https://github.com/catchorg/Catch2.git
            ${CATCH_COMMIT}
            ON
            )
    endif()

    endif()
endif()
