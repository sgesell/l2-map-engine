# L2MapEngine

A C++17 library for remapping finite element fields between meshes using $L^2$ projection. Given a scalar (or multi-component) field defined at integration points on a source mesh, L2MapEngine computes the best-fit projection onto a target mesh by solving local $L^2$-norm minimisation problems element-by-element.

## Features

- **2D and 3D support** — Quad4, Quad8 (2D) and Hex8, Tet4 (3D) elements
- **Exact polygon/polyhedron clipping** — Sutherland-Hodgman algorithm for mesh intersection
- **Stokes-theorem integration** — efficient polynomial integration over clipped regions without quadrature
- **Vandermonde-based basis construction** — numerically stable Lagrange bases via Eigen `FullPivLU`
- **BVH spatial indexing** — fast bounding-volume hierarchy for element-intersection queries
- **OpenMP parallelisation** — embarrassingly parallel element loop; falls back gracefully if OpenMP is unavailable
- **Python bindings** — thin NumPy-friendly wrapper via pybind11

## Method

For each new element $e_n$ the library solves the local system

$$\mathbf{V} \boldsymbol{\alpha} = \mathbf{M}$$

$$V_{ij} = \int_{e_n} \varphi_i \, \varphi_j \, dA$$

$$M_{jl} = \sum_{\substack{e_o \,:\, e_o \cap e_n \neq \emptyset}} \int_{e_o \cap e_n} \varphi_j \left( \sum_k \beta_{kl} \, \psi_k \right) dA$$

where $\varphi_i$ are the new element basis polynomials, $\psi_k$ the old element bases, and $\beta_{kl}$ the source field coefficients. See [docs/math_notes.md](docs/math_notes.md) for the full derivation.

## Requirements

| Dependency | Version | Notes |
|-----------|---------|-------|
| CMake | ≥ 3.18 | |
| C++ compiler | C++17 | MSVC 2019+, GCC 9+, Clang 10+ |
| Eigen3 | 3.4.0 | fetched automatically if not found |
| Catch2 | 3.5.2 | fetched automatically, tests only |
| pybind11 | 2.13.6 | fetched automatically, Python bindings only |
| OpenMP | any | optional — single-threaded fallback if absent |

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

To disable the Python bindings or tests:

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DL2MAP_BUILD_PYTHON=OFF \
    -DL2MAP_BUILD_TESTS=OFF
cmake --build build
```

### Running tests

```bash
cd build
ctest --output-on-failure
```

Tests use Catch2 tags to organise by subsystem:

| Tag | Scope |
|-----|-------|
| `[basis_builder]` | Vandermonde basis construction and Lagrange delta properties |
| `[mapping_engine]` | 2D mapping: identity, parallel, bounds enforcement, reference data |
| `[mapping_3d]` | 3D mapping: linear/quadratic fields, multi-component, parallel |
| `[mapping_3d][scale]` | 3D scale tests (1K, 10K fine elements) |
| `[poly_integrator]` | Stokes-theorem polygon integration |
| `[polygon_clipper]` | Sutherland-Hodgman clipping |
| `[reference]` | Regression tests against stored reference data |

Run a subset with:

```bash
cd build
ctest -R mapping_3d --output-on-failure
```

## Python usage

After building, the extension module `l2map_py` is in `build/python/` (or `build/python/Release/` on MSVC).

### 2D example (Quad8)

```python
import sys
sys.path.insert(0, "build/python/Release")  # adjust for your platform/config
import l2map_py
import numpy as np

# nodes:    shape (N, 3)       — columns: [id, x, y]     (1-indexed IDs)
# elements: shape (M, 9)       — columns: [id, n1..n8]   (1-indexed, Quad8)
# field:    shape (M*9, 2+K)   — columns: [elem_id, ipt_id, v1..vK]

result = l2map_py.map_integration_points(
    nodes_new, elements_new,
    nodes_old, elements_old,
    field_data,
    element_type="Quad8",
    enforce_positive=False,
    n_threads=-1,          # -1 = all available cores
)

print(result.values.shape)        # (M_new * 9, K)
print(len(result.ipoint_coords))  # M_new * 9 — list of (x, y) arrays
```

### 3D example (Hex8)

```python
# nodes:    shape (N, 4)       — columns: [id, x, y, z]  (1-indexed IDs)
# elements: shape (M, 9)       — columns: [id, n1..n8]   (1-indexed, Hex8)
# field:    shape (M*8, 2+K)   — columns: [elem_id, ipt_id, v1..vK]

result = l2map_py.map_integration_points(
    nodes_new, elements_new,
    nodes_old, elements_old,
    field_data,
    element_type="Hex8",   # dispatches to MappingEngine3D
    n_threads=-1,
)

print(result.values.shape)        # (M_new * 8, K)
print(len(result.ipoint_coords))  # M_new * 8 — list of (x, y, z) arrays
```

Supported `element_type` values: `"Quad4"`, `"Quad8"` (2D), `"Hex8"`, `"Tet4"` (3D).

## C++ usage

### 2D

```cpp
#include "l2map/mapping_engine.hpp"
#include "l2map/mesh.hpp"

l2map::MappingOptions opts;
opts.n_threads = 4;

l2map::MappingEngine engine(opts);
l2map::MappingResult result = engine.map_integration_points(old_mesh, new_mesh, field_data);
// result.values  — Eigen MatrixXd (n_new_ipts × n_components)
```

### 3D

```cpp
#include "l2map/mapping_engine_3d.hpp"
#include "l2map/mesh.hpp"

l2map::MappingOptions3D opts;
opts.n_threads = 4;

l2map::MappingEngine3D engine(opts);
l2map::MappingResult3D result = engine.map_integration_points(
    old_mesh, new_mesh, field_data, "Hex8");
// result.values        — Eigen MatrixXd (n_new_ipts × n_components)
// result.ipoint_coords — std::vector<Point3D>
```

## Project structure

```
include/l2map/        Public headers
src/                  Library implementation
python/               pybind11 bindings
tests/                Catch2 unit tests + reference data
tools/                Python reference implementation and test helpers
docs/                 Mathematical notes
cmake/                CMake dependency helpers
```

## References

- Gesell, S. (2024). "Anwendung des CTOD-Konzepts auf Rissfortschritt unter thermomechanischer Beanspruchung mithilfe von Experimenten und numerischer Simulation." Doctoral thesis, Technische Universität Bergakademie Freiberg. URN: [urn:nbn:de:bsz:105-qucosa2-928920](https://nbn-resolving.org/urn:nbn:de:bsz:105-qucosa2-928920)
- Chin, E.B., Lasserre, J.B., Sukumar, N. (2015). "Numerical integration of homogeneous functions on convex and nonconvex polygons and polyhedra." *Computational Mechanics*.
- ABAQUS Theory Manual — element formulations.

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for noncommercial use.
For commercial licensing, see [COMMERCIAL.md](COMMERCIAL.md).
