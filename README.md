# L2MapEngine

A C++17 library for remapping finite element fields between meshes using $L^2$ projection. Given a scalar (or multi-component) field defined at integration points on a source mesh, L2MapEngine computes the best-fit projection onto a target mesh by solving local $L^2$-norm minimisation problems element-by-element.

## Features

- **2D and 3D support** — Quad4, Quad8 (2D) and Hex8, Tet4 (3D) elements
- **Two 3D projection engines** — approximate (Gauss-quadrature, fast) and exact (polyhedral intersection, conservative)
- **Exact polygon/polyhedron clipping** — Sutherland-Hodgman algorithm for mesh intersection
- **Stokes-theorem integration** — exact polynomial integration over clipped regions without quadrature
- **Vandermonde-based basis construction** — numerically stable Lagrange bases via Eigen `ColPivHouseholderQR`
- **BVH spatial indexing** — fast bounding-volume hierarchy for element-intersection queries
- **OpenMP parallelisation** — embarrassingly parallel element loop; falls back gracefully if OpenMP is unavailable
- **Python bindings** — thin NumPy-friendly wrapper via pybind11

## Method

Both 3D engines solve the same local $L^2$ system per new element $e_n$:

$$\mathbf{V} \boldsymbol{\alpha} = \mathbf{M}$$

$$V_{ij} = \int_{e_n} \varphi_i \, \varphi_j \, dV, \qquad M_{jl} = \sum_{\substack{e_o \,:\, e_o \cap e_n \neq \emptyset}} \int_{e_o \cap e_n} \varphi_j \left( \sum_k \beta_{kl} \, \psi_k \right) dV$$

where $\varphi_i$ are the new element basis polynomials, $\psi_k$ the old element bases, and $\beta_{kl}$ the source field coefficients. They differ in **how** these integrals are evaluated.

### Approximate engine (`MappingEngine3D`)

Uses higher-order Gauss quadrature (27-point, $3^3$) in the physical domain of $e_n$.

**Mass matrix** — assembled by quadrature sum:

$$V_{ij} \approx \sum_q w_q \, \varphi_i(\mathbf{x}_q) \, \varphi_j(\mathbf{x}_q)$$

**RHS** — for each quadrature point $\mathbf{x}_q$, the old element containing $\mathbf{x}_q$ is found via BVH + axis-aligned bounding box containment, and the old field is evaluated by pointwise polynomial reconstruction:

$$M_{jl} \approx \sum_q w_q \, \varphi_j(\mathbf{x}_q) \, u_{\mathrm{old}}(\mathbf{x}_q)$$

where $u_{\mathrm{old}}(\mathbf{x}_q) = \sum_k \beta_{kl} \, \psi_k(\mathbf{x}_q)$ is evaluated from the old element's Lagrange basis.

No polyhedral geometry is computed. This makes the approximate engine significantly faster, and exact for trilinear fields on conforming or smoothly-overlapping meshes. Accuracy degrades near mesh boundaries where a quadrature point may straddle two old elements or fall outside the old mesh entirely (the contribution of that point is silently dropped).

### Exact engine (`MappingEngine3D_Exact`)

Computes both integrals exactly by:

1. **Polyhedral intersection** — the overlap $e_o \cap e_n$ is computed exactly via 3D Sutherland-Hodgman clipping.
2. **Divergence-theorem integration** — the polynomial integrand is integrated over the clipped polyhedron without introducing quadrature error, using the reduction

$$\int_P g \, dV = \sum_{d} \frac{1}{d+3} \sum_{\text{faces } k} (\mathbf{n}_k \cdot \mathbf{v}_{0k}) \int_{\text{face}_k} g_d \, dA$$

where $g_d$ is the degree-$d$ homogeneous part of $g$, $\mathbf{n}_k$ the outward face normal, and $\mathbf{v}_{0k}$ any vertex on that face.

This guarantees conservation of the field integral across any mesh pair, regardless of conformity. The cost is the geometric work (clipping + face integrals), which makes it slower than the approximate engine — roughly 5–20× depending on mesh complexity.

### Choosing an engine

| Criterion | Approximate | Exact |
|-----------|:-----------:|:-----:|
| Conforming / smoothly-overlapping meshes | ✓ (recommended) | ✓ |
| Non-conforming meshes with partial overlaps | acceptable | ✓ (recommended) |
| Integral conservation guaranteed | — | ✓ |
| Linear fields reproduced exactly | ✓ | ✓ |
| Speed | fast | ~5–20× slower |

See [docs/math_notes.md](docs/math_notes.md) for the full derivation.

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
| `[mapping_3d]` | 3D approximate engine: linear/quadratic fields, multi-component, parallel |
| `[mapping_3d][scale]` | 3D approximate scale tests (1K, 10K, 100K fine elements) |
| `[mapping_3d_exact]` | 3D exact engine: linear fields, multi-component, exact vs approx |
| `[mapping_3d_exact][scale]` | 3D exact scale tests (1K, 10K, 100K fine elements) |
| `[poly_integrator]` | Stokes-theorem polygon integration (2D) |
| `[poly_integrator_3d]` | Divergence-theorem polyhedron integration and polynomial multiplication (3D) |
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

### 3D example — approximate engine (Hex8)

The approximate engine is the default for 3D. It uses a 27-point Gauss quadrature rule
over each new element and evaluates the old field by pointwise polynomial reconstruction.
It is fast and accurate for conforming or smoothly-overlapping meshes.

```python
# nodes:    shape (N, 4)       — columns: [id, x, y, z]  (1-indexed IDs)
# elements: shape (M, 9)       — columns: [id, n1..n8]   (1-indexed, Hex8)
# field:    shape (M*8, 2+K)   — columns: [elem_id, ipt_id, v1..vK]

result = l2map_py.map_integration_points(
    nodes_new, elements_new,
    nodes_old, elements_old,
    field_data,
    element_type="Hex8",   # dispatches to MappingEngine3D (approximate)
    n_threads=-1,          # -1 = all available cores
)

print(result.values.shape)        # (M_new * 8, K)
print(len(result.ipoint_coords))  # M_new * 8 — list of (x, y, z) tuples
```

### 3D example — exact engine (Hex8)

The exact engine computes the polyhedral intersection $e_o \cap e_n$ via Sutherland-Hodgman
clipping and integrates over it exactly using the divergence theorem. Use this when
meshes are non-conforming or when integral conservation is required.

```python
result = l2map_py.map_integration_points(
    nodes_new, elements_new,
    nodes_old, elements_old,
    field_data,
    element_type="Hex8",
    method="exact",        # selects MappingEngine3D_Exact
    n_gauss_1d=5,          # Gauss points per edge for face integrals (default 5)
    n_threads=-1,
)

print(result.values.shape)        # (M_new * 8, K)
print(len(result.ipoint_coords))  # M_new * 8 — list of (x, y, z) tuples
```

Supported `element_type` values: `"Quad4"`, `"Quad8"` (2D), `"Hex8"`, `"Tet4"` (3D).  
The `method` parameter is `"approximate"` (default) or `"exact"` and only applies to 3D.

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
