#pragma once

#include "mesh.hpp"
namespace vem::two {
// reinitializes the centers as the centroids of the mesh elements
void set_centroids_as_centers(VEMMesh2 &vem);
}// namespace vem
