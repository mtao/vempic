#pragma once

#include "vem/mesh.hpp"
namespace vem {
// reinitializes the centers as the centroids of the mesh elements
void set_centroids_as_centers(VEMMesh2 &vem);
}// namespace vem