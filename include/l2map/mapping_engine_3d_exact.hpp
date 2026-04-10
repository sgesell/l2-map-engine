#pragma once
#include "types.hpp"
#include "mesh.hpp"
#include "bvh.hpp"
#include "basis_builder_3d.hpp"
#include "element_library_3d.hpp"
#include "polyhedron_clipper.hpp"
#include "poly_integrator_3d.hpp"
#include "parallel_executor.hpp"
#include <vector>

namespace l2map {

// ---------------------------------------------------------------------------
// MappingEngine3D_Exact — exact L² projection for 3D hexahedral meshes.
//
// Algorithm (mirrors the 2D engine exactly, extended to 3D):
//   For each new element e_n:
//     1. Build basis φ_i from the 8 physical integration points (Vandermonde).
//     2. Build e_n as a convex Polyhedron from its 8 node coordinates.
//     3. Mass matrix V[i,j] = ∫_{e_n} φ_i φ_j dV
//           — exact via divergence theorem on e_n (PolyIntegrator3D)
//     4. RHS M[j,l] = Σ_{e_o overlapping e_n}
//                       ∫_{e_o ∩ e_n} φ_j · u_old dV
//           — exact via polyhedral intersection (PolyhedronClipper)
//              + exact integration over intersection (PolyIntegrator3D)
//     5. Solve V · α = M
//
// This is the true L² projection including exact handling of partial overlaps.
// ---------------------------------------------------------------------------

struct MappingOptions3D_Exact {
    int  n_threads  = -1;
    int  n_gauss_1d = 5;   // 1D Gauss points for Stokes edge integrals
    bool verbose    = false;
};

class MappingEngine3D_Exact {
public:
    explicit MappingEngine3D_Exact(
        const MappingOptions3D_Exact& opts = MappingOptions3D_Exact{});

    // Same interface as MappingEngine3D.
    // field_data: (n_elem_old * 8, 2 + n_components)
    //   col 0 = elem ID (1-indexed), col 1 = ipt index (1-indexed)
    struct MappingResult3D_Exact {
        MatrixXd           values;
        std::vector<Point3D> ipoint_coords;
    };

    MappingResult3D_Exact map_integration_points(
        const Mesh& old_mesh,
        const Mesh& new_mesh,
        const MatrixXd& field_data,
        const std::string& element_type = "Hex8",
        const std::vector<ElemID>& elem_set_new = {});

private:
    MappingOptions3D_Exact opts_;
    BasisBuilder3D         basis_builder_;
    PolyhedronClipper      clipper_;
    // Single integrator instance shared across all element calls.
    // Its product-basis cache is pre-warmed before the parallel loop
    // so that parallel threads only perform reads (thread-safe).
    PolyIntegrator3D       integrator_;

    IntPointCache3D precompute_integration_points_(
        const Mesh& mesh, const std::string& type) const;

    BasisCache3D precompute_basis_matrices_(
        const Mesh& mesh,
        const IntPointCache3D& itp_cache) const;

    FieldDataCache build_field_cache_(const MatrixXd& field_data) const;

    MatrixXd map_single_element_(
        ElemID new_eid,
        const Mesh& new_mesh,
        const Mesh& old_mesh,
        const BVHTree3D& bvh,
        const IntPointCache3D& itp_new,
        const IntPointCache3D& itp_old,
        const BasisCache3D& basis_new,
        const BasisCache3D& basis_old,
        const FieldDataCache& field_cache,
        int n_components) const;

    // Build mass matrix V[i,j] = ∫_{poly_new} φ_i φ_j dV  (exact)
    MatrixXd build_mass_matrix_(const BasisMatrix& basis_new,
                                 const Polyhedron& poly_new,
                                 const MonomialBasis3D& mono,
                                 int N) const;

    // Build RHS M[j,l] from overlapping old elements (exact)
    MatrixXd build_rhs_(const BasisMatrix& basis_new,
                         const Polyhedron& poly_new,
                         const std::vector<ElemID>& candidates,
                         const Mesh& old_mesh,
                         const IntPointCache3D& itp_old,
                         const BasisCache3D& basis_old,
                         const FieldDataCache& field_cache,
                         const Point3D& shift,
                         const MonomialBasis3D& mono,
                         int N, int n_components) const;

    MatrixXd solve_system_(const MatrixXd& V, const MatrixXd& M) const;
};

} // namespace l2map
