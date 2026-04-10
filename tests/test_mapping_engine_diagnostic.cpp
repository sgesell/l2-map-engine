#define _USE_MATH_DEFINES
#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "l2map/mapping_engine.hpp"
#include "l2map/element_library.hpp"
#include "l2map/mesh.hpp"
#include "l2map/basis_builder.hpp"
#include "l2map/poly_integrator.hpp"
#include "l2map/polygon_clipper.hpp"
#include <vector>
#include <functional>

using namespace l2map;

// ===========================================================================
// Helpers
// ===========================================================================

// Build a regular nx×ny grid of Quad8 elements on [0, Lx] × [0, Ly].
// Returns Mesh with 0-indexed IDs.
static Mesh make_quad8_mesh(int nx, int ny, double Lx = 1.0, double Ly = 1.0) {
    double hx = Lx / nx;
    double hy = Ly / ny;
    int npts_x = 2 * nx + 1;
    int npts_y = 2 * ny + 1;

    std::vector<Node> nodes;
    NodeID nid = 0;
    for (int j = 0; j < npts_y; ++j)
        for (int i = 0; i < npts_x; ++i) {
            Node nd;
            nd.id = nid++;
            nd.x = i * hx * 0.5;
            nd.y = j * hy * 0.5;
            nodes.push_back(nd);
        }

    auto nidx = [&](int ci, int cj, int di, int dj) -> NodeID {
        return static_cast<NodeID>((cj + dj) * npts_x + (ci + di));
    };

    std::vector<Element> elements;
    ElemID eid = 0;
    for (int ej = 0; ej < ny; ++ej)
        for (int ei = 0; ei < nx; ++ei) {
            int ci = 2 * ei, cj = 2 * ej;
            Element e;
            e.id = eid++;
            e.type_name = "Quad8";
            e.node_ids = {
                nidx(ci,cj,0,0), nidx(ci,cj,2,0),
                nidx(ci,cj,2,2), nidx(ci,cj,0,2),
                nidx(ci,cj,1,0), nidx(ci,cj,2,1),
                nidx(ci,cj,1,2), nidx(ci,cj,0,1)
            };
            elements.push_back(e);
        }
    return Mesh(nodes, elements, "Quad8");
}

// Build field data from a callable f(x,y) on a Quad8 mesh.
// Returns shape (n_elem * 9, 2 + n_comp) with 1-indexed IDs.
template<typename F>
static MatrixXd make_field(const Mesh& mesh, F f) {
    const auto& eids = mesh.element_ids();
    int n_elem = static_cast<int>(eids.size());
    MatrixXd fd(n_elem * 9, 3);
    int row = 0;
    for (ElemID eid : eids) {
        MatrixXd coords = mesh.element_node_coords(eid);
        auto gpts = ElementLibrary::instance()
                        .compute_gauss_points_global("Quad8", coords);
        for (int ipt = 0; ipt < 9; ++ipt) {
            fd(row, 0) = static_cast<double>(eid + 1);
            fd(row, 1) = static_cast<double>(ipt + 1);
            fd(row, 2) = f(gpts[ipt][0], gpts[ipt][1]);
            ++row;
        }
    }
    return fd;
}

// Map and return max/rms error against exact f(x,y).
template<typename F>
static std::pair<double, double> map_and_measure_error(
    const Mesh& old_mesh, const Mesh& new_mesh,
    const MatrixXd& field_data,
    F f_exact)
{
    MappingOptions opts;
    opts.n_threads = 1;
    opts.verbose = false;
    MappingEngine engine(opts);
    MappingResult res = engine.map_integration_points(old_mesh, new_mesh, field_data);

    double max_err = 0.0, sum_sq = 0.0;
    int n = static_cast<int>(res.values.rows());
    for (int i = 0; i < n; ++i) {
        double exact = f_exact(res.ipoint_coords[i][0], res.ipoint_coords[i][1]);
        double err = std::abs(res.values(i, 0) - exact);
        max_err = std::max(max_err, err);
        sum_sq += err * err;
    }
    return {max_err, std::sqrt(sum_sq / n)};
}


// ===========================================================================
// DIAGNOSTIC TEST 1:
//   BasisBuilder reconstruction — interpolate a known polynomial at the
//   Gauss points and verify we can reproduce it at an arbitrary point.
//
//   This tests get_serendipity_basis_2d + BasisBuilder::build in isolation.
// ===========================================================================
TEST_CASE("Diagnostic: BasisBuilder reproduces quadratic field exactly",
          "[diagnostic][basis]")
{
    // One Quad8 element at [0,1]×[0,1]
    Mesh mesh = make_quad8_mesh(1, 1);
    ElemID eid = 0;
    MatrixXd coords = mesh.element_node_coords(eid);
    auto gpts = ElementLibrary::instance()
                    .compute_gauss_points_global("Quad8", coords);
    int N = static_cast<int>(gpts.size());
    REQUIRE(N == 9);

    // Quadratic field: f(x,y) = x^2 + 2xy + 3y^2
    auto f = [](double x, double y) { return x*x + 2*x*y + 3*y*y; };

    // Build basis
    BasisBuilder bb;
    BasisMatrix B = bb.build(gpts);
    MonomialBasis2D mono = get_serendipity_basis_2d(N);
    Point2D origin = gpts.back();

    // Field values at Gauss points
    VectorXd fvals(N);
    for (int i = 0; i < N; ++i)
        fvals[i] = f(gpts[i][0], gpts[i][1]);

    // Reconstruct at a set of test points within the element
    double test_pts[][2] = {{0.2, 0.3}, {0.5, 0.5}, {0.8, 0.1}, {0.1, 0.9}};
    for (auto& tp : test_pts) {
        Point2D p_sh(tp[0] - origin[0], tp[1] - origin[1]);
        VectorXd m = mono.evaluate(p_sh[0], p_sh[1]);
        double reconstructed = 0.0;
        for (int i = 0; i < N; ++i)
            reconstructed += fvals[i] * B.row(i).dot(m);
        double exact = f(tp[0], tp[1]);
        CHECK_THAT(reconstructed, Catch::Matchers::WithinAbs(exact, 1e-10));
    }
}

// ===========================================================================
// DIAGNOSTIC TEST 2:
//   Basis in a shifted frame — verify that building a Vandermonde directly
//   in a shifted frame gives the same interpolation as BasisBuilder::build
//   applied to already-shifted points.
//
//   This is the core of the original bug: BasisBuilder::build() shifts
//   points internally by pts.back(). If you pass already-shifted points,
//   you get a DOUBLE shift.
// ===========================================================================
TEST_CASE("Diagnostic: double-shift detection in BasisBuilder",
          "[diagnostic][basis][critical]")
{
    // One Quad8 element
    Mesh mesh = make_quad8_mesh(1, 1);
    ElemID eid = 0;
    MatrixXd coords = mesh.element_node_coords(eid);
    auto gpts_global = ElementLibrary::instance()
                           .compute_gauss_points_global("Quad8", coords);
    int N = static_cast<int>(gpts_global.size());

    // External shift (simulates new element's origin shift)
    Point2D ext_shift(0.3, 0.4);

    // Approach A: shift points, then pass to BasisBuilder::build()
    // build() will ADDITIONALLY subtract pts.back() internally → double shift
    std::vector<Point2D> pts_shifted(N);
    for (int i = 0; i < N; ++i)
        pts_shifted[i] = gpts_global[i] - ext_shift;

    BasisBuilder bb;
    BasisMatrix B_double = bb.build(pts_shifted);
    // The internal origin for B_double is pts_shifted.back() = gpts_global.back() - ext_shift
    // So monomials are evaluated at (p - ext_shift) - (gpts_global.back() - ext_shift)
    //                              = p - gpts_global.back()
    // That's in the WRONG frame for someone expecting coordinates relative to ext_shift.

    // Approach B: build Vandermonde directly (no extra shift) — correct approach
    MonomialBasis2D mono = get_serendipity_basis_2d(N);
    MatrixXd A_direct(N, N);
    for (int i = 0; i < N; ++i)
        A_direct.row(i) = mono.evaluate(pts_shifted[i][0], pts_shifted[i][1]);
    Eigen::FullPivLU<MatrixXd> lu(A_direct);
    REQUIRE(lu.rank() == N);
    MatrixXd B_direct = lu.inverse().transpose();  // same convention as BasisBuilder

    // Test: reconstruct f(x,y) = x^2 + 2xy + 3y^2 at a test point
    auto f = [](double x, double y) { return x*x + 2*x*y + 3*y*y; };

    VectorXd fvals(N);
    for (int i = 0; i < N; ++i)
        fvals[i] = f(gpts_global[i][0], gpts_global[i][1]);

    // Test point in shifted coordinates
    Point2D test_global(0.6, 0.7);
    Point2D test_shifted = test_global - ext_shift;
    VectorXd m_shifted = mono.evaluate(test_shifted[0], test_shifted[1]);

    // Approach B (direct Vandermonde): should be correct
    double recon_direct = 0.0;
    for (int i = 0; i < N; ++i)
        recon_direct += fvals[i] * B_direct.row(i).dot(m_shifted);
    double exact = f(test_global[0], test_global[1]);

    INFO("Direct Vandermonde reconstruction error: " << std::abs(recon_direct - exact));
    CHECK_THAT(recon_direct, Catch::Matchers::WithinAbs(exact, 1e-10));

    // Approach A (build()): uses double-shifted frame
    // To evaluate consistently with build()'s internal shift, we'd need to shift
    // the test point by pts_shifted.back(), not use test_shifted directly.
    // If someone naively uses test_shifted, they get the WRONG answer.
    Point2D internal_origin = pts_shifted.back();
    MonomialBasis2D mono_a = get_serendipity_basis_2d(N);

    // WRONG evaluation (as would happen in the buggy build_rhs_):
    double recon_wrong = 0.0;
    for (int i = 0; i < N; ++i)
        recon_wrong += fvals[i] * B_double.row(i).dot(m_shifted);

    // CORRECT evaluation (accounting for build()'s internal shift):
    Point2D test_double_shifted = test_shifted - internal_origin;
    VectorXd m_double = mono_a.evaluate(test_double_shifted[0], test_double_shifted[1]);
    double recon_correct = 0.0;
    for (int i = 0; i < N; ++i)
        recon_correct += fvals[i] * B_double.row(i).dot(m_double);

    INFO("build() with correct double-shift eval: " << std::abs(recon_correct - exact));
    CHECK_THAT(recon_correct, Catch::Matchers::WithinAbs(exact, 1e-10));

    // The wrong evaluation should NOT match (unless the function is linear/constant)
    double wrong_err = std::abs(recon_wrong - exact);
    INFO("build() with WRONG single-shift eval error: " << wrong_err);
    CHECK(wrong_err > 1e-4);  // expect significant error for a quadratic field
}


// ===========================================================================
// DIAGNOSTIC TEST 3:
//   Mapping a quadratic field exactly.
//   Quad8 (9 Gauss pts, biquadratic basis) should reproduce any function
//   in the polynomial space exactly. Uses f(x,y) = x^2 + 2xy + 3y^2.
//   Errors should be at machine precision.
// ===========================================================================
TEST_CASE("Diagnostic: quadratic field maps exactly (Quad8)",
          "[diagnostic][mapping][critical]")
{
    auto f = [](double x, double y) { return x*x + 2*x*y + 3*y*y; };

    Mesh old_mesh = make_quad8_mesh(3, 3);
    Mesh new_mesh = make_quad8_mesh(5, 5);

    MatrixXd fd = make_field(old_mesh, f);
    auto [max_err, rms_err] = map_and_measure_error(old_mesh, new_mesh, fd, f);

    INFO("Quadratic field max error: " << max_err);
    INFO("Quadratic field RMS error: " << rms_err);
    CHECK(max_err < 1e-8);
}


// ===========================================================================
// DIAGNOSTIC TEST 4:
//   Mapping a biquadratic field x^2 * y^2 — at the boundary of what
//   the Quad8 serendipity basis can represent.
// ===========================================================================
TEST_CASE("Diagnostic: biquadratic x2y2 field (Quad8)",
          "[diagnostic][mapping]")
{
    auto f = [](double x, double y) { return x*x * y*y; };

    Mesh old_mesh = make_quad8_mesh(4, 4);
    Mesh new_mesh = make_quad8_mesh(6, 6);

    MatrixXd fd = make_field(old_mesh, f);
    auto [max_err, rms_err] = map_and_measure_error(old_mesh, new_mesh, fd, f);

    INFO("Biquadratic field max error: " << max_err);
    INFO("Biquadratic field RMS error: " << rms_err);
    CHECK(max_err < 1e-8);
}


// ===========================================================================
// DIAGNOSTIC TEST 5:
//   Self-mapping (mesh to itself) with a nonlinear field.
//   The old and new meshes are IDENTICAL, so every new element maps to
//   exactly one old element with full overlap. The solution should be
//   exactly the input field values (no approximation error).
// ===========================================================================
TEST_CASE("Diagnostic: self-map identity for sin/cos field",
          "[diagnostic][mapping][critical]")
{
    auto f = [](double x, double y) {
        return std::sin(2 * M_PI * x) * std::cos(2 * M_PI * y);
    };

    Mesh mesh = make_quad8_mesh(5, 5);
    MatrixXd fd = make_field(mesh, f);

    MappingOptions opts;
    opts.n_threads = 1;
    MappingEngine engine(opts);
    MappingResult res = engine.map_integration_points(mesh, mesh, fd);

    // Compare mapped values to the INPUT field values (not to f(x,y)),
    // since the L2 projection on the same mesh should reproduce the
    // input exactly.
    double max_err = 0.0;
    int n = static_cast<int>(res.values.rows());
    for (int i = 0; i < n; ++i) {
        double err = std::abs(res.values(i, 0) - fd(i, 2));
        max_err = std::max(max_err, err);
    }
    INFO("Self-map max error: " << max_err);
    CHECK(max_err < 1e-8);
}


// ===========================================================================
// DIAGNOSTIC TEST 6:
//   GOLD STANDARD: verify that the mapping engine output exactly matches
//   the source polynomial evaluated at the target Gauss points.
//
//   For nested meshes where each target element is fully inside one source
//   element, the L2 projection of a Q2 polynomial onto a Q2 target space
//   must be exact.  Therefore:  mapped_value[i] == source_poly(target_pt[i])
//
//   This test is the definitive check for correctness.
// ===========================================================================
TEST_CASE("Diagnostic: mapped values match source polynomial evaluation",
          "[diagnostic][mapping][critical]")
{
    auto f = [](double x, double y) {
        return std::sin(2 * M_PI * x) * std::cos(2 * M_PI * y);
    };

    // Source: 3×3. Target: 6×6 (nested 2x refinement).
    Mesh src = make_quad8_mesh(3, 3);
    Mesh dst = make_quad8_mesh(6, 6);

    MatrixXd fd = make_field(src, f);

    // --- Run the mapping engine ---
    MappingOptions opts;
    opts.n_threads = 1;
    MappingEngine engine(opts);
    MappingResult res = engine.map_integration_points(src, dst, fd);

    // --- Independently evaluate the source polynomials at target points ---
    // For each source element, build the interpolating polynomial in GLOBAL
    // coordinates and evaluate it at the target Gauss points that lie within.

    // Build source element polynomials
    MonomialBasis2D mono = get_serendipity_basis_2d(9);
    std::unordered_map<ElemID, VectorXd> src_poly_coeffs;

    for (ElemID eid : src.element_ids()) {
        MatrixXd coords = src.element_node_coords(eid);
        auto gpts = ElementLibrary::instance()
                        .compute_gauss_points_global("Quad8", coords);
        MatrixXd A(9, 9);
        for (int i = 0; i < 9; ++i)
            A.row(i) = mono.evaluate(gpts[i][0], gpts[i][1]);

        // Field values for this element
        VectorXd fvals(9);
        for (int row = 0; row < fd.rows(); ++row) {
            ElemID eid_check = static_cast<ElemID>(std::round(fd(row, 0))) - 1;
            if (eid_check == eid) {
                int ipt = static_cast<int>(std::round(fd(row, 1))) - 1;
                fvals[ipt] = fd(row, 2);
            }
        }
        src_poly_coeffs[eid] = A.fullPivLu().solve(fvals);
    }

    // For each target point, find the containing source element and evaluate
    double max_diff = 0.0;
    int n_checked = 0;
    for (int i = 0; i < static_cast<int>(res.values.rows()); ++i) {
        Point2D p(res.ipoint_coords[i][0], res.ipoint_coords[i][1]);

        // Find the source element containing this point
        for (ElemID eid : src.element_ids()) {
            auto bbox = src.element_bbox(eid);
            double eps = 1e-10;
            if (p[0] >= bbox[0] - eps && p[0] <= bbox[1] + eps &&
                p[1] >= bbox[2] - eps && p[1] <= bbox[3] + eps) {
                // Evaluate source polynomial at this point
                VectorXd m = mono.evaluate(p[0], p[1]);
                double expected = src_poly_coeffs[eid].dot(m);
                double mapped = res.values(i, 0);
                double diff = std::abs(mapped - expected);
                max_diff = std::max(max_diff, diff);
                ++n_checked;
                break;
            }
        }
    }

    INFO("Checked " << n_checked << " / " << res.values.rows() << " points");
    INFO("Max diff (engine vs direct polynomial eval): " << max_diff);
    CHECK(n_checked == static_cast<int>(res.values.rows()));
    CHECK(max_diff < 1e-8);
}


// ===========================================================================
// DIAGNOSTIC TEST 7:
//   Non-nested meshes: map between meshes with different element counts
//   where elements partially overlap. This is the real stress test.
//   Uses sin/cos field and checks that the error is reasonable.
// ===========================================================================
TEST_CASE("Diagnostic: sin/cos field — non-nested meshes",
          "[diagnostic][mapping][critical]")
{
    auto f = [](double x, double y) {
        return std::sin(2 * M_PI * x) * std::cos(2 * M_PI * y);
    };

    // 5×5 → 7×7 (non-nested: element boundaries don't align)
    Mesh src = make_quad8_mesh(5, 5);
    Mesh dst = make_quad8_mesh(7, 7);

    MatrixXd fd_src = make_field(src, f);
    auto [max_err, rms_err] = map_and_measure_error(src, dst, fd_src, f);

    INFO("Non-nested 5→7 max error: " << max_err);
    INFO("Non-nested 5→7 RMS error: " << rms_err);

    // The source mesh resolves sin(2πx) reasonably well on a 5×5 grid.
    // Max error should be dominated by the source discretization error,
    // NOT by a systematic coordinate-frame bug.
    CHECK(max_err < 0.5);  // very loose — just catch catastrophic failures

    // Now test convergence: 10→14 should be strictly better than 5→7
    Mesh src2 = make_quad8_mesh(10, 10);
    Mesh dst2 = make_quad8_mesh(14, 14);
    MatrixXd fd_src2 = make_field(src2, f);
    auto [max_err2, rms_err2] = map_and_measure_error(src2, dst2, fd_src2, f);

    INFO("Non-nested 10→14 max error: " << max_err2);
    INFO("Non-nested 10→14 RMS error: " << rms_err2);
    CHECK(max_err2 < max_err);
    CHECK(rms_err2 < rms_err);
}


// ===========================================================================
// DIAGNOSTIC TEST 8:
//   Single-element overlap test.
//   Map from a 1×1 element to itself — isolates build_rhs_ + build_mass_matrix_
//   without BVH or multi-element complexity.
// ===========================================================================
TEST_CASE("Diagnostic: single-element self-map",
          "[diagnostic][mapping]")
{
    auto f = [](double x, double y) { return x*x + x*y + y*y; };

    Mesh mesh = make_quad8_mesh(1, 1);
    MatrixXd fd = make_field(mesh, f);

    MappingOptions opts;
    opts.n_threads = 1;
    MappingEngine engine(opts);
    MappingResult res = engine.map_integration_points(mesh, mesh, fd);

    REQUIRE(res.values.rows() == 9);
    double max_err = 0.0;
    for (int i = 0; i < 9; ++i) {
        double err = std::abs(res.values(i, 0) - fd(i, 2));
        max_err = std::max(max_err, err);
    }
    INFO("Single-element self-map max error: " << max_err);
    CHECK(max_err < 1e-10);
}


// ===========================================================================
// DIAGNOSTIC TEST 9:
//   Two-element overlap: map from 1×1 element to a 2×1 mesh (2 elements).
//   The source element covers both target elements.
//   Quadratic field — should be exact.
// ===========================================================================
TEST_CASE("Diagnostic: one source covering two target elements",
          "[diagnostic][mapping]")
{
    auto f = [](double x, double y) { return x*x + 2*x*y + y*y; };

    // Source: 1×1 on [0,1]×[0,1]
    Mesh src = make_quad8_mesh(1, 1);
    // Target: 2×1 on [0,1]×[0,1] (two elements side by side)
    Mesh dst = make_quad8_mesh(2, 1);

    MatrixXd fd = make_field(src, f);
    auto [max_err, rms_err] = map_and_measure_error(src, dst, fd, f);

    INFO("1-to-2 elements, quadratic field, max error: " << max_err);
    CHECK(max_err < 1e-8);
}


// ===========================================================================
// DIAGNOSTIC TEST 10:
//   Source-refinement convergence for sin/cos.
//   Refine the SOURCE mesh (keep target proportional).
//   Error should decrease at ≥ O(h^2) for Quad8.
// ===========================================================================
TEST_CASE("Diagnostic: convergence rate for sin/cos (source refinement)",
          "[diagnostic][mapping][convergence]")
{
    auto f = [](double x, double y) {
        return std::sin(2 * M_PI * x) * std::cos(2 * M_PI * y);
    };

    int configs[][2] = {{4, 6}, {8, 12}, {16, 24}};
    double prev_max = 1e30;
    double prev_h = 0.0;

    for (auto& cfg : configs) {
        int n_src = cfg[0], n_dst = cfg[1];
        Mesh src = make_quad8_mesh(n_src, n_src);
        Mesh dst = make_quad8_mesh(n_dst, n_dst);
        MatrixXd fd = make_field(src, f);
        auto [max_err, rms_err] = map_and_measure_error(src, dst, fd, f);
        double h = 1.0 / n_src;

        INFO("  source " << n_src << "x" << n_src
             << " → target " << n_dst << "x" << n_dst
             << ": h=" << h << " max=" << max_err << " rms=" << rms_err);

        if (prev_h > 0) {
            double rate = std::log(prev_max / max_err) / std::log(prev_h / h);
            INFO("  convergence rate: " << rate);
            // Expect at least O(h^2) for Quad8
            CHECK(rate > 1.5);
        }
        CHECK(max_err < prev_max);  // error must decrease
        prev_max = max_err;
        prev_h = h;
    }
}


// ===========================================================================
// DIAGNOSTIC TEST 11:
//   Linear field mapping — must be exact regardless of mesh pair.
//   Tests the most basic correctness requirement.
// ===========================================================================
TEST_CASE("Diagnostic: linear field is exact on any mesh pair",
          "[diagnostic][mapping][critical]")
{
    auto f = [](double x, double y) { return 1.0 + 2.0*x + 3.0*y; };

    struct Config {
        int src_n, dst_n;
    };
    Config configs[] = {{3, 5}, {4, 7}, {6, 3}, {2, 9}};

    for (auto& c : configs) {
        Mesh src = make_quad8_mesh(c.src_n, c.src_n);
        Mesh dst = make_quad8_mesh(c.dst_n, c.dst_n);
        MatrixXd fd = make_field(src, f);
        auto [max_err, rms_err] = map_and_measure_error(src, dst, fd, f);

        INFO("Linear field " << c.src_n << "→" << c.dst_n
             << ": max=" << max_err << " rms=" << rms_err);
        CHECK(max_err < 1e-8);
    }
}
