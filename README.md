# L2MapEngine

A C++17 library for remapping finite element fields between meshes using L² projection. Given a scalar (or multi-component) field defined at integration points on a source mesh, L2MapEngine computes the best-fit projection onto a target mesh by solving local L²-norm minimisation problems element-by-element.

## Features

- **2D and 3D support** — Quad8 (8-node serendipity quad) and Hex20 (20-node serendipity hex) elements
- **Exact polygon/polyhedron clipping** — Sutherland-Hodgman algorithm for mesh intersection
- **Stokes-theorem integration** — efficient polynomial integration over clipped regions without quadrature
- **Vandermonde-based basis construction** — numerically stable Lagrange bases via Eigen `FullPivLU`
- **BVH spatial indexing** — fast bounding-volume hierarchy for element-intersection queries
- **OpenMP parallelisation** — embarrassingly parallel element loop; falls back gracefully if OpenMP is unavailable
- **Python bindings** — thin NumPy-friendly wrapper via pybind11

## Method

For each new element `e_n` the library solves the local system

```
V · α = M

V[i,j] = ∫_{e_n} φ_i φ_j dA
M[j,l]  = Σ_{e_o ∩ e_n ≠ ∅}  ∫_{e_o ∩ e_n} φ_j · (Σ_k β_{k,l} ψ_k) dA
```

where `φ` are the new element basis polynomials, `ψ` the old element bases, and `β` the source field coefficients. See [docs/math_notes.md](docs/math_notes.md) for the full derivation.

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

## Python usage

After building, the extension module `l2map_py` is in `build/python/` (or `build/python/Release/` on MSVC).

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

print(result.values.shape)   # (M_new * n_ipts, K)
```

## C++ usage

```cpp
#include "l2map/mapping_engine.hpp"
#include "l2map/mesh.hpp"

l2map::MappingOptions opts;
opts.n_threads = 4;

l2map::MappingEngine engine(opts);
l2map::MappingResult result = engine.map_integration_points(old_mesh, new_mesh, field_data);
// result.values  — Eigen MatrixXd (n_new_ipts × n_components)
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
