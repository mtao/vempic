# VEMPIC
A reference implementation for VEMPIC, a fluid simulator that combines cut-cell meshes, the Virtual Element Method, and the PIC framework.
This library uses [Mandoline](github.com/mtao/mandoline) to construct its cut-cell meshes, so please make sure you can build Mandoline's dependencies as well.

## Organization
There is a core set of tools in the root `./include` and `./src` directories for VEM.
If you're interested in conformal VEM look at the `Point*` code, if you're interested in non-Conformal VEM look at the `Flux*` code.
In the `./applications` diretory there are a number of examples applications of VEM.

In Two dimensions:
* poisson - a poisson equation solver, GUI includes some fun advection tools IIRC
* wavesim - a semi-implicit wave simulator
* fluidsim - the two dimensional implementation of VEMPIC
In Three dimensions:
* fluidsim - the full three dimensional VEMPIC


## Status
The code, as it is, buildable, but requires some extra work to run the examples because I need to document the input format and change some hardcoded output directories.
I'll also be switching to [meson](https://mesonbuild.com) instead of cmake, which will result in substantial changes on the dependencies held in Mandoline.
For now please treat this codebase as a reference rather than as an implementation to build upon.
As I plan on using this codebase for future projects it should naturally be cleaned up over time, just as Mandoline underwent some feature development and cleanup to make this project work.

## Compilation
VEMPIC depends on a C++20 enabled compiler.
Recently I have only tried compiling with the following:
- [gcc](https://gcc.gnu.org) with version = 11.3.0 
- [cmake](https://cmake.org) with version = 3.19.7 .
On Ubuntu the following should build
```bash
apt install git cmake build-essential gcc
```
I prefer using [Ninja](https://ninja-build.org/) build system over make, so the following will use the `ninja` command instead of `make`.


```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja
ninja
```

### Build Issues
As it is, this code has a number of dependencies that are somewhat fickle to build.
Here's a few tips.

#### cutmesh.pb.h not found
Sometimes CMake fails to build the protobuf header `cutmesh.pb.h` before building Mandoline sources, which depend on it.
This can be explicitly built with
```bash
ninja cutmesh_proto
```
in the build directory.

#### Embree undefined references
On my AMD Zen3 Linux machine I had difficulty building Embree without disabling some ISA. My solution was to disable some ISA:
```
${CMAKE_CMDS} -DEMBREE_MAX_ISA=NONE -DEMBREE_ISA_SSE2=OFF -DEMBREE_ISA_SSE42=OFF -DEMBREE_ISA_AVX=ON -DEMBREE_ISA_AVX2=ON -DEMBREE_ISA_AVX412=OFF 
```
If you're using AMD and are using Excavator or any of hte Zen series processors the above should work.
If you are using intel you're on a relatively recent Intel processor (Skylake or later) but not too new (Alder Lake with new microcode) then this step can probably be ignored...
