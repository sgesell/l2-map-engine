#pragma once
#include "types.hpp"
#include "mesh.hpp"
#include "bvh.hpp"
#include "basis_builder.hpp"
#include "polygon_clipper.hpp"
#include "poly_integrator.hpp"
#include "parallel_executor.hpp"
#include <vector>

namespace l2map {

struct MappingOptions {
    bool enforce_bounds   = false; // clip result to [min, max] of source values
    bool enforce_positive = false; // clip result to [0, ∞)
    int  n_gauss_pts      = 5;     // Gauss points per edge in integrator
    int  n_threads        = -1;    // -1 = all available cores
    bool verbose          = false;
};

struct MappingResult {
    // Shape: (n_elements_new * n_ipts_per_element, n_components)
    // Rows ordered: element 0 pts 0..N-1, element 1 pts 0..N-1, ...
    MatrixXd values;

    // Integration point global coordinates
    std::vector<Point2D> ipoint_coords;

    // Per-element count of clipped values (when bounds enforcement is on)
    std::vector<int> n_clipped;
};

class MappingEngine {
public:
    explicit MappingEngine(const MappingOptions& opts = MappingOptions{});

    // Map integration point data from old mesh to new mesh.
    //
    // field_data: shape (n_elements_old * n_ipts, 2 + n_components)
    //   col 0 = element ID (1-indexed), col 1 = ipt index (1-indexed),
    //   cols 2..end = field values
    //
    // elem_set_new: 0-indexed element IDs to map into; empty = all elements
    MappingResult map_integration_points(
        const Mesh& old_mesh,
        const Mesh& new_mesh,
        const MatrixXd& field_data,
        const std::vector<ElemID>& elem_set_new = {});

    // Map nodal data — Phase 1 implementation uses simple L2 element-local approach.
    MappingResult map_nodal_data(
        const Mesh& old_mesh,
        const Mesh& new_mesh,
        const MatrixXd& field_data,
        const std::vector<NodeID>& node_set_new = {});

private:
    MappingOptions    opts_;
    BasisBuilder      basis_builder_;
    PolygonClipper    clipper_;

    // Precompute global integration point coordinates for all elements in mesh
    IntPointCache precompute_integration_points_(const Mesh& mesh) const;

    // Build basis matrix for each element (points already shifted to origin = last Gauss pt)
    BasisCache precompute_basis_matrices_(const Mesh& mesh,
                                          const IntPointCache& itp_cache) const;

    // Organise field_data rows into per-element cache
    // field_data cols: [elem_id(1-idx), ipt_id(1-idx), v0, v1, ...]
    FieldDataCache build_field_cache_(const MatrixXd& field_data) const;

    // Per-element mapping (called inside parallel loop)
    MatrixXd map_single_element_(
        ElemID new_elem_id,
        const Mesh& new_mesh,
        const Mesh& old_mesh,
        const BVHTree2D& bvh,
        const IntPointCache& itp_new,
        const IntPointCache& itp_old,
        const BasisCache& basis_new,
        const FieldDataCache& field_cache,
        int n_components) const;

    // Build the local mass matrix V (N×N)
    MatrixXd build_mass_matrix_(
        const BasisMatrix& basis,
        const Polygon2D& elem_poly_shifted,
        const MonomialBasis2D& mono,
        int N) const;

    // Build the RHS M (N × n_components)
    MatrixXd build_rhs_(
        const BasisMatrix& basis_new,
        const Polygon2D& new_poly,
        const std::vector<ElemID>& overlap_ids,
        const Mesh& old_mesh,
        const IntPointCache& itp_old,
        const FieldDataCache& field_cache,
        const Point2D& shift,
        const MonomialBasis2D& mono,
        int N, int n_components) const;

    // Solve V * alpha = M; apply optional bounds enforcement
    MatrixXd solve_system_(const MatrixXd& V, const MatrixXd& M,
                            double vmin = 0.0, double vmax = 0.0) const;

    // Shift all polygon vertices by -shift
    static Polygon2D shift_polygon_(const Polygon2D& poly, const Point2D& shift);
};

} // namespace l2map
