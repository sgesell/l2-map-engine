#pragma once
#include "types.hpp"
#include "mesh.hpp"
#include "bvh.hpp"
#include "basis_builder_3d.hpp"
#include "element_library_3d.hpp"
#include "parallel_executor.hpp"
#include <vector>

namespace l2map {

struct MappingOptions3D {
    int  n_threads = -1;    // -1 = all available cores
    bool verbose   = false;
};

struct MappingResult3D {
    // Shape: (n_elements_new * n_ipts_per_element, n_components)
    MatrixXd values;

    // Physical integration point coordinates (n_ipts_total x 3 not stored — use ipoint_coords)
    std::vector<Point3D> ipoint_coords;
};

// ---------------------------------------------------------------------------
// MappingEngine3D — L² projection for 3D hexahedral meshes.
//
// Algorithm (Gauss-quadrature variational transfer):
//   For each new element e_n:
//     1. Build basis φ_i from the 8 physical integration points of e_n.
//     2. Compute 27-point higher-order quadrature in physical space (with Jacobian).
//     3. Mass matrix V[i,j] = Σ_q w_q * φ_i(x_q - shift) * φ_j(x_q - shift)
//     4. RHS M[j,l] = Σ_q w_q * φ_j(x_q - shift) * u_old(x_q, l)
//        where u_old(x_q) is evaluated from the old element containing x_q.
//     5. Solve V * α = M  →  α = mapped field values at new element integration points.
//
// field_data format: (n_elem_old * n_ipts, 2 + n_components)
//   col 0 = element ID (1-indexed), col 1 = integration point index (1-indexed)
//   cols 2.. = field component values
// ---------------------------------------------------------------------------

class MappingEngine3D {
public:
    explicit MappingEngine3D(const MappingOptions3D& opts = MappingOptions3D{});

    MappingResult3D map_integration_points(
        const Mesh& old_mesh,
        const Mesh& new_mesh,
        const MatrixXd& field_data,
        const std::string& element_type = "Hex8",
        const std::vector<ElemID>& elem_set_new = {});

private:
    MappingOptions3D  opts_;
    BasisBuilder3D    basis_builder_;

    // Precompute physical Gauss points for all elements
    IntPointCache3D precompute_integration_points_(
        const Mesh& mesh, const std::string& type) const;

    // Precompute basis matrices from physical Gauss points (shifted, last = origin)
    BasisCache3D precompute_basis_matrices_(
        const Mesh& mesh,
        const IntPointCache3D& itp_cache,
        const std::string& type) const;

    // Precompute higher-order quadrature points + physical weights per new element
    void precompute_quad_points_(
        const Mesh& mesh,
        const std::string& type,
        QuadPointCache3D& quad_pts_out,
        QuadWeightCache3D& quad_wts_out) const;

    // Parse field_data into per-element cache
    FieldDataCache build_field_cache_(const MatrixXd& field_data) const;

    // Map one new element; returns alpha matrix (n_ipts x n_components)
    MatrixXd map_single_element_(
        ElemID new_eid,
        const Mesh& new_mesh,
        const Mesh& old_mesh,
        const BVHTree3D& bvh,
        const IntPointCache3D& itp_new,
        const IntPointCache3D& itp_old,
        const BasisCache3D& basis_new,
        const BasisCache3D& basis_old,
        const QuadPointCache3D& quad_pts,
        const QuadWeightCache3D& quad_wts,
        const FieldDataCache& field_cache,
        int n_components) const;

    // Evaluate old field at a physical point x using old element's basis
    // Returns VectorXd of length n_components (zeros if not found)
    VectorXd eval_old_field_(
        const Point3D& x,
        ElemID old_eid,
        const IntPointCache3D& itp_old,
        const BasisCache3D& basis_old,
        const FieldDataCache& field_cache,
        int n_components) const;

    // Solve V * alpha = M with fallback
    MatrixXd solve_system_(const MatrixXd& V, const MatrixXd& M) const;
};

} // namespace l2map
