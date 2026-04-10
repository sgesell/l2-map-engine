#include "l2map/mapping_engine_3d_exact.hpp"
#include "l2map/element_library_3d.hpp"
#include <iostream>
#include <stdexcept>

namespace l2map {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MappingEngine3D_Exact::MappingEngine3D_Exact(const MappingOptions3D_Exact& opts)
    : opts_(opts), integrator_(opts.n_gauss_1d)
{}

// Workaround: PolyIntegrator3D is created per map call to avoid header complexity.
// Actual implementation uses PolyIntegrator3D directly.

// ---------------------------------------------------------------------------
// Precomputation (identical to approximate engine)
// ---------------------------------------------------------------------------

IntPointCache3D MappingEngine3D_Exact::precompute_integration_points_(
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

BasisCache3D MappingEngine3D_Exact::precompute_basis_matrices_(
    const Mesh& mesh, const IntPointCache3D& itp_cache) const
{
    BasisCache3D cache;
    for (ElemID eid : mesh.element_ids()) {
        const std::vector<Point3D>& pts = itp_cache.at(eid);
        Point3D origin = pts.back();
        std::vector<Point3D> shifted(pts.size());
        for (size_t i = 0; i < pts.size(); ++i)
            shifted[i] = pts[i] - origin;
        cache[eid] = basis_builder_.build(shifted);
    }
    return cache;
}

FieldDataCache MappingEngine3D_Exact::build_field_cache_(const MatrixXd& field_data) const {
    FieldDataCache cache;
    int nc = static_cast<int>(field_data.cols()) - 2;
    for (int row = 0; row < field_data.rows(); ++row) {
        ElemID eid = static_cast<ElemID>(std::round(field_data(row, 0))) - 1;
        int    ipt = static_cast<int>   (std::round(field_data(row, 1))) - 1;
        auto it = cache.find(eid);
        if (it == cache.end()) { cache[eid] = MatrixXd(); it = cache.find(eid); }
        FieldSlice& fs = it->second;
        if (ipt >= fs.rows()) fs.conservativeResize(std::max(ipt+1,(int)fs.rows()+1), nc);
        fs.row(ipt) = field_data.row(row).tail(nc);
    }
    return cache;
}

// ---------------------------------------------------------------------------
// Mass matrix: V[i,j] = ∫_{poly_new} φ_i φ_j dV  — exact integration
// ---------------------------------------------------------------------------

MatrixXd MappingEngine3D_Exact::build_mass_matrix_(
    const BasisMatrix& basis_new,
    const Polyhedron& poly_new,
    const MonomialBasis3D& mono,
    int N) const
{
    MatrixXd V = MatrixXd::Zero(N, N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            MonomialBasis3D mono_prod;
            VectorXd prod = integrator_.multiply_polynomials(
                basis_new.row(i), mono, basis_new.row(j), mono, mono_prod);
            double val = integrator_.integrate(poly_new, prod, mono_prod);
            V(i, j) = val;
            V(j, i) = val;
        }
    }
    return V;
}

// ---------------------------------------------------------------------------
// RHS: M[j,l] = Σ_{eₒ} ∫_{eₒ ∩ eₙ} φ_j · u_old dV  — exact
// ---------------------------------------------------------------------------

MatrixXd MappingEngine3D_Exact::build_rhs_(
    const BasisMatrix& basis_new,
    const Polyhedron& poly_new,
    const std::vector<ElemID>& candidates,
    const Mesh& old_mesh,
    const IntPointCache3D& itp_old,
    const BasisCache3D& basis_old,
    const FieldDataCache& field_cache,
    const Point3D& shift,
    const MonomialBasis3D& mono,
    int N, int n_components) const
{
    MatrixXd M = MatrixXd::Zero(N, n_components);

    // All old elements in a uniform Hex8 mesh have the same number of integration
    // points — hoist the monomial basis construction outside the candidate loop so
    // it is built once instead of once per overlapping element.
    if (candidates.empty()) return M;
    int N_old = static_cast<int>(itp_old.begin()->second.size());
    MonomialBasis3D mono_old = get_tensor_basis_3d(N_old);

    for (ElemID oid : candidates) {
        auto fc_it = field_cache.find(oid);
        if (fc_it == field_cache.end()) continue;
        const FieldSlice& beta = fc_it->second;   // (n_ipts × n_comp)

        // Build old element as polyhedron in UN-shifted physical coords
        MatrixXd old_coords = old_mesh.element_node_coords_3d(oid);
        Polyhedron poly_old = hex8_to_polyhedron(old_coords);

        // Compute intersection (both polys in original physical coords)
        auto isect_opt = clipper_.intersect(poly_new, poly_old);
        if (!isect_opt) continue;
        const Polyhedron& isect = *isect_opt;

        // Reject near-zero-volume intersections
        double vol = std::abs(clipper_.signed_volume(isect));
        if (vol < 1e-30) continue;

        // Shift intersection into new element's local coordinate system
        // (subtract `shift`, the last Gauss point of the new element)
        Polyhedron isect_sh;
        isect_sh.faces.reserve(isect.faces.size());
        for (const auto& face : isect.faces) {
            Face3D f_sh;
            f_sh.reserve(face.size());
            for (const auto& v : face) f_sh.push_back(v - shift);
            isect_sh.faces.push_back(std::move(f_sh));
        }

        // Old element's Gauss points in the NEW element's shifted frame (y = x - shift)
        const std::vector<Point3D>& itp_old_global = itp_old.at(oid);

        std::vector<Point3D> itp_old_y(N_old);
        for (int k = 0; k < N_old; ++k)
            itp_old_y[k] = itp_old_global[k] - shift;

        // Build Vandermonde in the y frame directly (no extra internal shift).
        // V_old[p][q] = mono_old_q(y_p)  where y_p = g_p^old - shift
        MatrixXd V_old(N_old, N_old);
        for (int p = 0; p < N_old; ++p)
            V_old.row(p) = mono_old.evaluate(itp_old_y[p][0], itp_old_y[p][1], itp_old_y[p][2]);

        // Solve V_old * field_poly = beta to get polynomial coefficients of u_old in y frame.
        // field_poly(:, l) = coefficients such that u_old(y) = mono_old(y)^T * field_poly(:,l)
        // PartialPivLU is faster than FullPivLU for the well-conditioned Vandermonde system.
        int n_comp_local = std::min(n_components, static_cast<int>(beta.cols()));
        int n_beta_rows  = std::min(N_old, static_cast<int>(beta.rows()));
        MatrixXd rhs = MatrixXd::Zero(N_old, n_comp_local);
        if (n_beta_rows > 0)
            rhs.topRows(n_beta_rows) = beta.topRows(n_beta_rows).leftCols(n_comp_local);
        MatrixXd field_poly = V_old.partialPivLu().solve(rhs);

        // For each test function φ_j and component l:
        //   M[j,l] += ∫_{isect_sh} φ_j(x) · field_poly[l](x) dV
        for (int j = 0; j < N; ++j) {
            VectorXd phi_j = basis_new.row(j);  // coeffs of φ_j in mono basis (shifted)
            for (int l = 0; l < n_comp_local; ++l) {
                VectorXd field_l = field_poly.col(l);  // coeffs of u_old^l in mono_old
                MonomialBasis3D mono_prod;
                VectorXd prod = integrator_.multiply_polynomials(
                    phi_j, mono, field_l, mono_old, mono_prod);
                M(j, l) += integrator_.integrate(isect_sh, prod, mono_prod);
            }
        }
    }
    return M;
}

// ---------------------------------------------------------------------------
// Solve V · α = M
// ---------------------------------------------------------------------------

MatrixXd MappingEngine3D_Exact::solve_system_(const MatrixXd& V,
                                               const MatrixXd& M) const
{
    Eigen::LLT<MatrixXd> llt(V);
    if (llt.info() == Eigen::Success) return llt.solve(M);
    return V.partialPivLu().solve(M);
}

// ---------------------------------------------------------------------------
// Per-element mapping
// ---------------------------------------------------------------------------

MatrixXd MappingEngine3D_Exact::map_single_element_(
    ElemID new_eid,
    const Mesh& new_mesh,
    const Mesh& old_mesh,
    const BVHTree3D& bvh,
    const IntPointCache3D& itp_new,
    const IntPointCache3D& itp_old,
    const BasisCache3D& basis_new,
    const BasisCache3D& basis_old,
    const FieldDataCache& field_cache,
    int n_components) const
{
    (void)basis_old;  // old basis rebuilt per candidate element in build_rhs_

    const std::vector<Point3D>& ipts_new = itp_new.at(new_eid);
    int N = static_cast<int>(ipts_new.size());  // 8 for Hex8

    // Shift: last Gauss point of new element = origin
    Point3D shift = ipts_new.back();

    MonomialBasis3D mono = get_tensor_basis_3d(N);
    const BasisMatrix& bn = basis_new.at(new_eid);

    // Build new element as a Polyhedron (unshifted, for intersection with old elements)
    MatrixXd new_coords = new_mesh.element_node_coords_3d(new_eid);
    Polyhedron poly_new_global = hex8_to_polyhedron(new_coords);

    // Shifted version (for mass matrix integration)
    Polyhedron poly_new_sh;
    poly_new_sh.faces.reserve(poly_new_global.faces.size());
    for (const auto& face : poly_new_global.faces) {
        Face3D f_sh;
        f_sh.reserve(face.size());
        for (const auto& v : face) f_sh.push_back(v - shift);
        poly_new_sh.faces.push_back(std::move(f_sh));
    }

    // Mass matrix (exact integration over shifted new element volume)
    MatrixXd V = build_mass_matrix_(bn, poly_new_sh, mono, N);

    // Find overlapping old elements via BVH
    AABB3D new_aabb = new_mesh.element_bbox_3d(new_eid);
    const double eps = 1e-10;
    new_aabb.xmin -= eps; new_aabb.xmax += eps;
    new_aabb.ymin -= eps; new_aabb.ymax += eps;
    new_aabb.zmin -= eps; new_aabb.zmax += eps;
    std::vector<ElemID> candidates = bvh.query_overlaps(new_aabb);

    // RHS (exact polyhedral intersection + integration)
    MatrixXd M = build_rhs_(bn, poly_new_global, candidates,
                              old_mesh, itp_old, basis_old, field_cache,
                              shift, mono, N, n_components);

    return solve_system_(V, M);
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

MappingEngine3D_Exact::MappingResult3D_Exact
MappingEngine3D_Exact::map_integration_points(
    const Mesh& old_mesh,
    const Mesh& new_mesh,
    const MatrixXd& field_data,
    const std::string& element_type,
    const std::vector<ElemID>& elem_set_new)
{
    std::vector<ElemID> elem_set = elem_set_new;
    if (elem_set.empty()) elem_set = new_mesh.element_ids();

    int n_components = static_cast<int>(field_data.cols()) - 2;
    if (n_components <= 0)
        throw std::invalid_argument("field_data must have at least 3 columns");

    if (opts_.verbose)
        std::cout << "[L2Map3D_Exact] Mapping " << elem_set.size()
                  << " elements, " << n_components << " components\n";

    // BVH for old mesh
    const auto& old_ids = old_mesh.element_ids();
    std::vector<AABB3D> old_bboxes;
    old_bboxes.reserve(old_ids.size());
    for (ElemID eid : old_ids) old_bboxes.push_back(old_mesh.element_bbox_3d(eid));
    BVHTree3D bvh;
    bvh.build(old_ids, old_bboxes);

    IntPointCache3D itp_new = precompute_integration_points_(new_mesh, element_type);
    IntPointCache3D itp_old = precompute_integration_points_(old_mesh, element_type);
    BasisCache3D basis_new  = precompute_basis_matrices_(new_mesh, itp_new);
    BasisCache3D basis_old  = precompute_basis_matrices_(old_mesh, itp_old);
    FieldDataCache field_cache = build_field_cache_(field_data);

    int n_ipts = static_cast<int>(itp_new.begin()->second.size());

    // Pre-warm the product-basis cache for the degree pair used by this element
    // type, so that parallel threads in the loop below only read from the cache
    // (concurrent reads on std::unordered_map are thread-safe in C++11).
    {
        MonomialBasis3D mono_sample = get_tensor_basis_3d(n_ipts);
        int deg = mono_sample.max_degree();
        integrator_.warm_up_product_cache(deg, deg);
    }
    int n_elem = static_cast<int>(elem_set.size());

    MatrixXd results = MatrixXd::Zero(n_elem * n_ipts, n_components);
    std::vector<Point3D> all_ipt_coords;
    all_ipt_coords.reserve(n_elem * n_ipts);
    for (int ei = 0; ei < n_elem; ++ei)
        for (const auto& pt : itp_new.at(elem_set[ei]))
            all_ipt_coords.push_back(pt);

    ParallelExecutor executor(opts_.n_threads);
    executor.parallel_for(n_elem, [&](int ei) {
        ElemID new_eid = elem_set[ei];
        MatrixXd alpha = map_single_element_(
            new_eid, new_mesh, old_mesh, bvh,
            itp_new, itp_old, basis_new, basis_old, field_cache, n_components);
        results.block(ei * n_ipts, 0, n_ipts, n_components) = alpha;
    });

    return {results, all_ipt_coords};
}

} // namespace l2map
