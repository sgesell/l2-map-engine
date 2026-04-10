#include "l2map/mapping_engine.hpp"
#include "l2map/element_library.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>

namespace l2map {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MappingEngine::MappingEngine(const MappingOptions& opts)
    : opts_(opts),
      clipper_()
{}

// ---------------------------------------------------------------------------
// Static helper
// ---------------------------------------------------------------------------

Polygon2D MappingEngine::shift_polygon_(const Polygon2D& poly, const Point2D& shift) {
    Polygon2D shifted(poly.size());
    for (size_t i = 0; i < poly.size(); ++i)
        shifted[i] = poly[i] - shift;
    return shifted;
}

// ---------------------------------------------------------------------------
// Precomputation
// ---------------------------------------------------------------------------

IntPointCache MappingEngine::precompute_integration_points_(const Mesh& mesh) const {
    IntPointCache cache;
    const std::string& type = mesh.default_type();
    for (ElemID eid : mesh.element_ids()) {
        MatrixXd coords = mesh.element_node_coords(eid);
        cache[eid] = ElementLibrary::instance().compute_gauss_points_global(type, coords);
    }
    return cache;
}

BasisCache MappingEngine::precompute_basis_matrices_(const Mesh& mesh,
                                                       const IntPointCache& itp_cache) const
{
    BasisCache cache;
    for (ElemID eid : mesh.element_ids()) {
        const std::vector<Point2D>& pts = itp_cache.at(eid);
        // Shift: subtract last point so it becomes origin (critical for conditioning)
        Point2D origin = pts.back();
        std::vector<Point2D> shifted(pts.size());
        for (size_t i = 0; i < pts.size(); ++i)
            shifted[i] = pts[i] - origin;
        cache[eid] = basis_builder_.build(shifted);
    }
    return cache;
}

FieldDataCache MappingEngine::build_field_cache_(const MatrixXd& field_data) const {
    // field_data columns: [elem_id(1-idx), ipt_id(1-idx), v0, v1, ...]
    FieldDataCache cache;
    int n_components = static_cast<int>(field_data.cols()) - 2;

    for (int row = 0; row < field_data.rows(); ++row) {
        ElemID eid   = static_cast<ElemID>(std::round(field_data(row, 0))) - 1; // → 0-indexed
        int    ipt   = static_cast<int>   (std::round(field_data(row, 1))) - 1; // → 0-indexed

        auto it = cache.find(eid);
        if (it == cache.end()) {
            // We don't know n_ipts ahead of time; resize lazily
            cache[eid] = MatrixXd();
            it = cache.find(eid);
        }

        FieldSlice& fs = it->second;
        if (ipt >= fs.rows()) {
            int new_rows = std::max(ipt + 1, static_cast<int>(fs.rows()) + 1);
            fs.conservativeResize(new_rows, n_components);
        }
        fs.row(ipt) = field_data.row(row).tail(n_components);
    }
    return cache;
}

// ---------------------------------------------------------------------------
// Mass matrix: V[i,j] = ∫_poly φ_i * φ_j dA   (symmetric)
// ---------------------------------------------------------------------------

MatrixXd MappingEngine::build_mass_matrix_(
    const BasisMatrix& basis,
    const Polygon2D& elem_poly_shifted,
    const MonomialBasis2D& mono,
    int N) const
{
    PolyIntegrator integrator(opts_.n_gauss_pts);
    MatrixXd V = MatrixXd::Zero(N, N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            MonomialBasis2D mono_prod;
            VectorXd prod = integrator.multiply_polynomials(
                basis.row(i), mono, basis.row(j), mono, mono_prod);
            double val = integrator.integrate(elem_poly_shifted, prod, mono_prod);
            V(i, j) = val;
            V(j, i) = val;  // symmetric
        }
    }
    return V;
}

// ---------------------------------------------------------------------------
// RHS: M[j,l] = Σ_{old overlaps} ∫_{intersection} φ_j * (Σ_i β_{i,l} ψ_i) dA
// ---------------------------------------------------------------------------

MatrixXd MappingEngine::build_rhs_(
    const BasisMatrix& basis_new,
    const Polygon2D& new_poly,
    const std::vector<ElemID>& overlap_ids,
    const Mesh& old_mesh,
    const IntPointCache& itp_old,
    const FieldDataCache& field_cache,
    const Point2D& shift,
    const MonomialBasis2D& mono,
    int N, int n_components) const
{
    PolyIntegrator integrator(opts_.n_gauss_pts);
    MatrixXd M = MatrixXd::Zero(N, n_components);

    for (ElemID oid : overlap_ids) {
        // Check if this element has field data
        auto fc_it = field_cache.find(oid);
        if (fc_it == field_cache.end()) continue;
        const FieldSlice& beta = fc_it->second; // shape (n_ipts, n_components)

        // Compute intersection in unshifted global coords
        Polygon2D poly_old = old_mesh.element_polygon(oid);
        auto intersection_opt = clipper_.intersect(poly_old, new_poly);
        if (!intersection_opt) continue;

        // Shift intersection into new element's local coordinate system
        Polygon2D intersection_sh = shift_polygon_(*intersection_opt, shift);

        // Rebuild old element basis in new element's coordinate system
        const std::vector<Point2D>& itp_old_global = itp_old.at(oid);
        std::vector<Point2D> itp_old_shifted(itp_old_global.size());
        for (size_t k = 0; k < itp_old_global.size(); ++k)
            itp_old_shifted[k] = itp_old_global[k] - shift;

        BasisMatrix basis_old_local = basis_builder_.build(itp_old_shifted);
        int N_old = static_cast<int>(itp_old_shifted.size());

        // Precompute field interpolation polynomials for each component:
        //   field_poly[:, l] = Σ_i beta[i,l] * basis_old_local[i, :]
        // Shape: (n_monomials, n_components)
        int n_comp_local = std::min(n_components, static_cast<int>(beta.cols()));
        MatrixXd field_poly = MatrixXd::Zero(N_old, n_comp_local);
        for (int i = 0; i < N_old; ++i) {
            if (i >= beta.rows()) break;
            for (int l = 0; l < n_comp_local; ++l)
                field_poly.col(l) += beta(i, l) * basis_old_local.row(i).transpose();
        }

        // For each test function j, integrate φ_j * field_poly[:, l] over intersection
        for (int j = 0; j < N; ++j) {
            VectorXd phi_j = basis_new.row(j);
            for (int l = 0; l < n_comp_local; ++l) {
                VectorXd field_l = field_poly.col(l);
                MonomialBasis2D mono_prod;
                VectorXd prod = integrator.multiply_polynomials(
                    phi_j, mono, field_l, mono, mono_prod);
                M(j, l) += integrator.integrate(intersection_sh, prod, mono_prod);
            }
        }
    }
    return M;
}

// ---------------------------------------------------------------------------
// Solve V * alpha = M
// ---------------------------------------------------------------------------

MatrixXd MappingEngine::solve_system_(const MatrixXd& V, const MatrixXd& M,
                                       double vmin, double vmax) const
{
    // Try Cholesky (V should be SPD)
    Eigen::LLT<MatrixXd> llt(V);
    MatrixXd alpha;
    if (llt.info() == Eigen::Success) {
        alpha = llt.solve(M);
    } else {
        // Fallback to partial pivot LU
        alpha = V.partialPivLu().solve(M);
    }

    if (opts_.enforce_positive) {
        alpha = alpha.cwiseMax(0.0);
    } else if (opts_.enforce_bounds && vmax > vmin) {
        alpha = alpha.cwiseMax(vmin).cwiseMin(vmax);
    }
    return alpha;
}

// ---------------------------------------------------------------------------
// Per-element mapping
// ---------------------------------------------------------------------------

MatrixXd MappingEngine::map_single_element_(
    ElemID new_elem_id,
    const Mesh& new_mesh,
    const Mesh& old_mesh,
    const BVHTree2D& bvh,
    const IntPointCache& itp_new,
    const IntPointCache& itp_old,
    const BasisCache& basis_new,
    const FieldDataCache& field_cache,
    int n_components) const
{
    const auto& itp_new_pts = itp_new.at(new_elem_id);
    int N = static_cast<int>(itp_new_pts.size());

    // Shift origin = last Gauss point of new element
    Point2D shift = itp_new_pts.back();

    // Query BVH candidates
    auto bbox_arr = new_mesh.element_bbox(new_elem_id);
    AABB2D query_box{bbox_arr[0], bbox_arr[1], bbox_arr[2], bbox_arr[3]};
    std::vector<ElemID> candidates = bvh.query_overlaps(query_box);

    // New element polygon (unshifted for intersection; shifted for V build)
    Polygon2D poly_new = new_mesh.element_polygon(new_elem_id);
    Polygon2D poly_new_sh = shift_polygon_(poly_new, shift);

    MonomialBasis2D mono = get_serendipity_basis_2d(N);
    const BasisMatrix& bn = basis_new.at(new_elem_id);

    // Mass matrix V
    MatrixXd V = build_mass_matrix_(bn, poly_new_sh, mono, N);

    // RHS M
    MatrixXd M = build_rhs_(bn, poly_new, candidates, old_mesh,
                              itp_old, field_cache, shift, mono, N, n_components);

    // Solve
    return solve_system_(V, M);
}

// ---------------------------------------------------------------------------
// Public: map_integration_points
// ---------------------------------------------------------------------------

MappingResult MappingEngine::map_integration_points(
    const Mesh& old_mesh,
    const Mesh& new_mesh,
    const MatrixXd& field_data,
    const std::vector<ElemID>& elem_set_new)
{
    // Determine element set
    std::vector<ElemID> elem_set = elem_set_new;
    if (elem_set.empty())
        elem_set = new_mesh.element_ids();

    int n_components = static_cast<int>(field_data.cols()) - 2;
    if (n_components <= 0)
        throw std::invalid_argument("field_data must have at least 3 columns");

    if (opts_.verbose)
        std::cout << "[L2Map] Mapping " << elem_set.size() << " new elements, "
                  << n_components << " components\n";

    // Step 1: Build BVH for old mesh
    const auto& old_ids = old_mesh.element_ids();
    std::vector<AABB2D> old_bboxes;
    old_bboxes.reserve(old_ids.size());
    for (ElemID eid : old_ids) {
        auto bb = old_mesh.element_bbox(eid);
        old_bboxes.push_back({bb[0], bb[1], bb[2], bb[3]});
    }
    BVHTree2D bvh;
    bvh.build(old_ids, old_bboxes);

    // Step 2-3: Precompute integration points for both meshes
    if (opts_.verbose) std::cout << "[L2Map] Precomputing integration points...\n";
    IntPointCache itp_new = precompute_integration_points_(new_mesh);
    IntPointCache itp_old = precompute_integration_points_(old_mesh);

    // Step 4: Precompute basis matrices for new mesh
    if (opts_.verbose) std::cout << "[L2Map] Precomputing basis matrices...\n";
    BasisCache basis_new = precompute_basis_matrices_(new_mesh, itp_new);

    // Step 5: Build field data cache
    FieldDataCache field_cache = build_field_cache_(field_data);

    // Get n_ipts from first element
    int n_ipts = static_cast<int>(itp_new.begin()->second.size());
    int n_elem = static_cast<int>(elem_set.size());

    // Allocate output
    MatrixXd results = MatrixXd::Zero(n_elem * n_ipts, n_components);
    std::vector<Point2D> all_ipt_coords;
    all_ipt_coords.reserve(n_elem * n_ipts);

    // Pre-fill ipt_coords (serial, before parallel loop)
    for (int ei = 0; ei < n_elem; ++ei) {
        const auto& pts = itp_new.at(elem_set[ei]);
        for (const auto& pt : pts)
            all_ipt_coords.push_back(pt);
    }

    if (opts_.verbose) std::cout << "[L2Map] Starting parallel element loop...\n";

    // Step 6: Parallel loop over new mesh elements
    ParallelExecutor executor(opts_.n_threads);
    executor.parallel_for(n_elem, [&](int ei) {
        ElemID new_eid = elem_set[ei];
        MatrixXd alpha = map_single_element_(
            new_eid, new_mesh, old_mesh, bvh,
            itp_new, itp_old, basis_new, field_cache, n_components);

        int row_start = ei * n_ipts;
        results.block(row_start, 0, n_ipts, n_components) = alpha;
    });

    MappingResult res;
    res.values       = results;
    res.ipoint_coords = all_ipt_coords;
    res.n_clipped.assign(n_elem, 0);
    return res;
}

// ---------------------------------------------------------------------------
// Public: map_nodal_data (Phase 1 stub — element-local L2 approach)
// ---------------------------------------------------------------------------

MappingResult MappingEngine::map_nodal_data(
    const Mesh& old_mesh,
    const Mesh& new_mesh,
    const MatrixXd& field_data,
    const std::vector<NodeID>& node_set_new)
{
    (void)old_mesh; (void)new_mesh; (void)field_data; (void)node_set_new;
    throw std::runtime_error("map_nodal_data: full L2 nodal mapping is Phase 2. "
                             "Use the SFM (Shape Function Method) from transform_nodes.py for Phase 1.");
}

} // namespace l2map
