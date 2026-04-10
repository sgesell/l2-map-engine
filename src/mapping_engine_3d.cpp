#include "l2map/mapping_engine_3d.hpp"
#include "l2map/element_library_3d.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace l2map {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MappingEngine3D::MappingEngine3D(const MappingOptions3D& opts)
    : opts_(opts)
{}

// ---------------------------------------------------------------------------
// Precomputation helpers
// ---------------------------------------------------------------------------

IntPointCache3D MappingEngine3D::precompute_integration_points_(
    const Mesh& mesh, const std::string& type) const
{
    IntPointCache3D cache;
    for (ElemID eid : mesh.element_ids()) {
        MatrixXd coords = mesh.element_node_coords_3d(eid);
        cache[eid] = ElementLibrary3D::instance()
                         .compute_gauss_points_global(type, coords);
    }
    return cache;
}

BasisCache3D MappingEngine3D::precompute_basis_matrices_(
    const Mesh& mesh,
    const IntPointCache3D& itp_cache,
    const std::string& /*type*/) const
{
    BasisCache3D cache;
    for (ElemID eid : mesh.element_ids()) {
        const std::vector<Point3D>& pts = itp_cache.at(eid);
        // Shift: subtract last point so it becomes the origin
        Point3D origin = pts.back();
        std::vector<Point3D> shifted(pts.size());
        for (size_t i = 0; i < pts.size(); ++i)
            shifted[i] = pts[i] - origin;
        cache[eid] = basis_builder_.build(shifted);
    }
    return cache;
}

void MappingEngine3D::precompute_quad_points_(
    const Mesh& mesh,
    const std::string& type,
    QuadPointCache3D& quad_pts_out,
    QuadWeightCache3D& quad_wts_out) const
{
    for (ElemID eid : mesh.element_ids()) {
        MatrixXd coords = mesh.element_node_coords_3d(eid);
        ElementLibrary3D::instance().compute_quad_points_global(
            type, coords, quad_pts_out[eid], quad_wts_out[eid]);
    }
}

FieldDataCache MappingEngine3D::build_field_cache_(const MatrixXd& field_data) const {
    FieldDataCache cache;
    int n_components = static_cast<int>(field_data.cols()) - 2;
    for (int row = 0; row < field_data.rows(); ++row) {
        ElemID eid = static_cast<ElemID>(std::round(field_data(row, 0))) - 1;
        int    ipt = static_cast<int>   (std::round(field_data(row, 1))) - 1;
        auto it = cache.find(eid);
        if (it == cache.end()) {
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
// Evaluate old field at physical point x from old element old_eid
// u_old(x) = Σ_p β_{p,l} * φ_p_old(x - shift_old)
// ---------------------------------------------------------------------------

VectorXd MappingEngine3D::eval_old_field_(
    const Point3D& x,
    ElemID old_eid,
    const IntPointCache3D& itp_old,
    const BasisCache3D& basis_old,
    const FieldDataCache& field_cache,
    int n_components) const
{
    auto fc_it = field_cache.find(old_eid);
    if (fc_it == field_cache.end())
        return VectorXd::Zero(n_components);

    const FieldSlice& beta = fc_it->second;  // (n_ipts x n_components)
    const std::vector<Point3D>& pts = itp_old.at(old_eid);
    const BasisMatrix& basis = basis_old.at(old_eid);

    // shift = last integration point of old element
    Point3D shift_old = pts.back();
    Point3D x_shifted = x - shift_old;

    int N_old = static_cast<int>(pts.size());
    MonomialBasis3D mono = get_tensor_basis_3d(N_old);
    VectorXd m = mono.evaluate(x_shifted[0], x_shifted[1], x_shifted[2]);

    int n_comp = std::min(n_components, static_cast<int>(beta.cols()));
    VectorXd result = VectorXd::Zero(n_components);

    // u_old(x, l) = Σ_p β_{p,l} * (basis[p, :] · m)
    for (int p = 0; p < N_old && p < beta.rows(); ++p) {
        double phi_p = basis.row(p).dot(m);
        for (int l = 0; l < n_comp; ++l)
            result[l] += beta(p, l) * phi_p;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Solve V * alpha = M
// ---------------------------------------------------------------------------

MatrixXd MappingEngine3D::solve_system_(const MatrixXd& V, const MatrixXd& M) const {
    Eigen::LLT<MatrixXd> llt(V);
    if (llt.info() == Eigen::Success)
        return llt.solve(M);
    return V.partialPivLu().solve(M);
}

// ---------------------------------------------------------------------------
// Per-element mapping
// ---------------------------------------------------------------------------

MatrixXd MappingEngine3D::map_single_element_(
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
    int n_components) const
{
    const std::vector<Point3D>& ipts_new = itp_new.at(new_eid);
    int N = static_cast<int>(ipts_new.size());  // 8 for Hex8

    // Shift: last Gauss point of new element = origin
    Point3D shift = ipts_new.back();

    MonomialBasis3D mono = get_tensor_basis_3d(N);
    const BasisMatrix& bn = basis_new.at(new_eid);  // (N x N): row i = φ_i coefficients

    // -----------------------------------------------------------------------
    // Step 1: Mass matrix V[i,j] = Σ_q w_q * φ_i(x_q - shift) * φ_j(x_q - shift)
    // -----------------------------------------------------------------------
    const std::vector<Point3D>& qpts = quad_pts.at(new_eid);
    const std::vector<double>&  qwts = quad_wts.at(new_eid);
    int nq = static_cast<int>(qpts.size());

    MatrixXd V = MatrixXd::Zero(N, N);
    MatrixXd M_rhs = MatrixXd::Zero(N, n_components);

    // Pre-evaluate φ_i at all quadrature points: phi_at_quad(i, q) = φ_i(x_q - shift)
    MatrixXd phi_at_quad(N, nq);
    for (int q = 0; q < nq; ++q) {
        Point3D xq_sh = qpts[q] - shift;
        VectorXd m = mono.evaluate(xq_sh[0], xq_sh[1], xq_sh[2]);
        phi_at_quad.col(q) = bn * m;
    }

    // Mass matrix V (symmetric)
    for (int q = 0; q < nq; ++q) {
        VectorXd phi_q = phi_at_quad.col(q);
        V += qwts[q] * phi_q * phi_q.transpose();
    }

    // -----------------------------------------------------------------------
    // Step 2: Find old elements overlapping new element via BVH.
    // Use node-based AABB (exact extent of the hex element).
    // -----------------------------------------------------------------------
    AABB3D new_aabb = new_mesh.element_bbox_3d(new_eid);
    const double eps = 1e-10;
    new_aabb.xmin -= eps; new_aabb.xmax += eps;
    new_aabb.ymin -= eps; new_aabb.ymax += eps;
    new_aabb.zmin -= eps; new_aabb.zmax += eps;

    std::vector<ElemID> candidates = bvh.query_overlaps(new_aabb);

    // -----------------------------------------------------------------------
    // Step 3: For each quadrature point, find the old element containing it
    //         using the node-based AABB (correct for axis-aligned hex).
    //         Accumulate RHS M.
    // -----------------------------------------------------------------------
    for (int q = 0; q < nq; ++q) {
        const Point3D& xq = qpts[q];

        ElemID old_eid = -1;
        for (ElemID cand : candidates) {
            // Node-based AABB is the exact extent of an axis-aligned hex element
            AABB3D cand_bbox = old_mesh.element_bbox_3d(cand);
            if (cand_bbox.contains(xq)) {
                old_eid = cand;
                break;
            }
        }

        if (old_eid < 0) continue;

        VectorXd u_q = eval_old_field_(xq, old_eid, itp_old, basis_old, field_cache, n_components);

        VectorXd phi_q = phi_at_quad.col(q);
        M_rhs += qwts[q] * phi_q * u_q.transpose();
    }

    // -----------------------------------------------------------------------
    // Step 4: Solve V * alpha = M
    // -----------------------------------------------------------------------
    return solve_system_(V, M_rhs);
}

// ---------------------------------------------------------------------------
// Public: map_integration_points
// ---------------------------------------------------------------------------

MappingResult3D MappingEngine3D::map_integration_points(
    const Mesh& old_mesh,
    const Mesh& new_mesh,
    const MatrixXd& field_data,
    const std::string& element_type,
    const std::vector<ElemID>& elem_set_new)
{
    std::vector<ElemID> elem_set = elem_set_new;
    if (elem_set.empty())
        elem_set = new_mesh.element_ids();

    int n_components = static_cast<int>(field_data.cols()) - 2;
    if (n_components <= 0)
        throw std::invalid_argument("field_data must have at least 3 columns");

    if (opts_.verbose)
        std::cout << "[L2Map3D] Mapping " << elem_set.size() << " new elements, "
                  << n_components << " components, type=" << element_type << "\n";

    // Step 1: Build BVH3D for old mesh
    if (opts_.verbose) std::cout << "[L2Map3D] Building BVH3D...\n";
    const auto& old_ids = old_mesh.element_ids();
    std::vector<AABB3D> old_bboxes;
    old_bboxes.reserve(old_ids.size());
    for (ElemID eid : old_ids)
        old_bboxes.push_back(old_mesh.element_bbox_3d(eid));
    BVHTree3D bvh;
    bvh.build(old_ids, old_bboxes);

    // Step 2: Precompute integration points (physical Gauss points)
    if (opts_.verbose) std::cout << "[L2Map3D] Precomputing integration points...\n";
    IntPointCache3D itp_new = precompute_integration_points_(new_mesh, element_type);
    IntPointCache3D itp_old = precompute_integration_points_(old_mesh, element_type);

    // Step 3: Precompute basis matrices from integration points
    if (opts_.verbose) std::cout << "[L2Map3D] Precomputing basis matrices...\n";
    BasisCache3D basis_new = precompute_basis_matrices_(new_mesh, itp_new, element_type);
    BasisCache3D basis_old = precompute_basis_matrices_(old_mesh, itp_old, element_type);

    // Step 4: Precompute higher-order quadrature (for computing V and M)
    if (opts_.verbose) std::cout << "[L2Map3D] Precomputing quadrature points...\n";
    QuadPointCache3D  quad_pts;
    QuadWeightCache3D quad_wts;
    precompute_quad_points_(new_mesh, element_type, quad_pts, quad_wts);

    // Step 5: Field data cache
    FieldDataCache field_cache = build_field_cache_(field_data);

    int n_ipts = static_cast<int>(itp_new.begin()->second.size());
    int n_elem = static_cast<int>(elem_set.size());

    MatrixXd results = MatrixXd::Zero(n_elem * n_ipts, n_components);
    std::vector<Point3D> all_ipt_coords;
    all_ipt_coords.reserve(n_elem * n_ipts);

    for (int ei = 0; ei < n_elem; ++ei) {
        const auto& pts = itp_new.at(elem_set[ei]);
        for (const auto& pt : pts)
            all_ipt_coords.push_back(pt);
    }

    if (opts_.verbose) std::cout << "[L2Map3D] Starting parallel element loop...\n";

    // Step 6: Parallel loop over new elements
    ParallelExecutor executor(opts_.n_threads);
    executor.parallel_for(n_elem, [&](int ei) {
        ElemID new_eid = elem_set[ei];
        MatrixXd alpha = map_single_element_(
            new_eid, new_mesh, old_mesh, bvh,
            itp_new, itp_old, basis_new, basis_old,
            quad_pts, quad_wts, field_cache, n_components);
        int row_start = ei * n_ipts;
        results.block(row_start, 0, n_ipts, n_components) = alpha;
    });

    MappingResult3D res;
    res.values        = results;
    res.ipoint_coords = all_ipt_coords;
    return res;
}

} // namespace l2map
