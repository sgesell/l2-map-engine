#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "l2map/mapping_engine_3d.hpp"
#include "l2map/mapping_engine_3d_exact.hpp"
#include "l2map/element_library_3d.hpp"
#include "l2map/basis_builder_3d.hpp"
#include "l2map/mesh.hpp"
#include <cmath>
#include <chrono>
#include <iostream>

using namespace l2map;

// =============================================================================
// Test geometry: regular axis-aligned Hex8 grid
//
// Mesh covers [0, Lx] x [0, Ly] x [0, Lz]
// with nx x ny x nz Hex8 elements.
//
// Node numbering: x fastest, then y, then z
//   Node(i,j,k) = k*(nx+1)*(ny+1) + j*(nx+1) + i
//
// Element numbering: same ijk ordering
//   Elem(ei,ej,ek) = ek*nx*ny + ej*nx + ei
//
// Hex8 node ordering (Abaqus):
//   Bottom face (z-): N0=BL, N1=BR, N2=TR, N3=TL
//   Top    face (z+): N4=BL, N5=BR, N6=TR, N7=TL
// =============================================================================

static Mesh make_hex8_mesh(int nx, int ny, int nz,
                            double Lx = 1.0, double Ly = 1.0, double Lz = 1.0)
{
    double hx = Lx / nx;
    double hy = Ly / ny;
    double hz = Lz / nz;

    // Nodes
    std::vector<Node> nodes;
    nodes.reserve((nx+1)*(ny+1)*(nz+1));
    NodeID nid = 0;
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i <= nx; ++i) {
                Node nd;
                nd.id = nid++;
                nd.x  = i * hx;
                nd.y  = j * hy;
                nd.z  = k * hz;
                nodes.push_back(nd);
            }

    auto node_id = [&](int i, int j, int k) -> NodeID {
        return static_cast<NodeID>(k*(nx+1)*(ny+1) + j*(nx+1) + i);
    };

    // Elements
    std::vector<Element> elements;
    elements.reserve(nx*ny*nz);
    ElemID eid = 0;
    for (int ek = 0; ek < nz; ++ek)
        for (int ej = 0; ej < ny; ++ej)
            for (int ei = 0; ei < nx; ++ei) {
                Element e;
                e.id = eid++;
                e.type_name = "Hex8";
                // Bottom face (z-): BL, BR, TR, TL
                // Top    face (z+): BL, BR, TR, TL
                e.node_ids = {
                    node_id(ei,   ej,   ek  ),  // N0
                    node_id(ei+1, ej,   ek  ),  // N1
                    node_id(ei+1, ej+1, ek  ),  // N2
                    node_id(ei,   ej+1, ek  ),  // N3
                    node_id(ei,   ej,   ek+1),  // N4
                    node_id(ei+1, ej,   ek+1),  // N5
                    node_id(ei+1, ej+1, ek+1),  // N6
                    node_id(ei,   ej+1, ek+1),  // N7
                };
                elements.push_back(e);
            }

    return Mesh(nodes, elements, "Hex8");
}

// Build field_data matrix (n_elem * 8, 2 + n_comp) where field = f(x,y,z)
// for each integration point of each element.
static MatrixXd make_field_data(const Mesh& mesh,
                                std::function<VectorXd(double, double, double)> f,
                                int n_comp)
{
    const auto& elem_ids = mesh.element_ids();
    int n_elem = static_cast<int>(elem_ids.size());
    MatrixXd fd(n_elem * 8, 2 + n_comp);

    int row = 0;
    for (ElemID eid : elem_ids) {
        MatrixXd coords = mesh.element_node_coords_3d(eid);
        auto gauss_pts = ElementLibrary3D::instance()
                             .compute_gauss_points_global("Hex8", coords);
        for (int ipt = 0; ipt < 8; ++ipt) {
            double x = gauss_pts[ipt][0];
            double y = gauss_pts[ipt][1];
            double z = gauss_pts[ipt][2];
            VectorXd val = f(x, y, z);
            fd(row, 0) = static_cast<double>(eid + 1);
            fd(row, 1) = static_cast<double>(ipt + 1);
            fd.row(row).tail(n_comp) = val;
            ++row;
        }
    }
    return fd;
}

// =============================================================================
// Test 1: Identity mapping (same mesh → same mesh)
// Field: linear f(x,y,z) = 1 + 2x + 3y + 4z
// Linear fields are exactly representable → max error should be ~machine epsilon
// =============================================================================
TEST_CASE("3D MappingEngine: identity linear field", "[mapping_3d]") {
    auto linear_f = [](double x, double y, double z) -> VectorXd {
        VectorXd v(1);
        v[0] = 1.0 + 2.0*x + 3.0*y + 4.0*z;
        return v;
    };

    // 4x4x4 = 64 elements
    Mesh mesh = make_hex8_mesh(4, 4, 4, 1.0, 1.0, 1.0);
    MatrixXd fd = make_field_data(mesh, linear_f, 1);

    MappingOptions3D opts;
    opts.n_threads = 1;
    MappingEngine3D engine(opts);

    MappingResult3D res = engine.map_integration_points(mesh, mesh, fd, "Hex8");

    REQUIRE(res.values.rows() == fd.rows());
    REQUIRE(res.values.cols() == 1);

    double max_err = 0.0;
    for (int i = 0; i < res.values.rows(); ++i) {
        double expected = fd(i, 2);
        double got      = res.values(i, 0);
        max_err = std::max(max_err, std::abs(got - expected));
    }
    INFO("Max error (linear field, identity): " << max_err);
    CHECK(max_err < 1e-8);
}

// =============================================================================
// Test 2: Coarse → Fine mapping, linear field
// Coarse: 3×3×3 = 27 elements
// Fine:   6×6×6 = 216 elements (2× refinement)
// The linear field is within both spaces → exact reproduction expected
// =============================================================================
TEST_CASE("3D MappingEngine: coarse-to-fine linear field", "[mapping_3d]") {
    auto linear_f = [](double x, double y, double z) -> VectorXd {
        VectorXd v(1);
        v[0] = 2.0 + 1.5*x - 0.5*y + 3.0*z;
        return v;
    };

    Mesh coarse = make_hex8_mesh(3, 3, 3, 1.0, 1.0, 1.0);
    Mesh fine   = make_hex8_mesh(6, 6, 6, 1.0, 1.0, 1.0);
    MatrixXd fd = make_field_data(coarse, linear_f, 1);

    MappingOptions3D opts;
    opts.n_threads = 1;
    MappingEngine3D engine(opts);

    MappingResult3D res = engine.map_integration_points(coarse, fine, fd, "Hex8");

    // Check against direct evaluation on fine mesh
    MatrixXd fd_fine = make_field_data(fine, linear_f, 1);

    double max_err = 0.0;
    for (int i = 0; i < res.values.rows(); ++i) {
        double expected = fd_fine(i, 2);
        double got      = res.values(i, 0);
        max_err = std::max(max_err, std::abs(got - expected));
    }
    INFO("Max error (linear, coarse->fine 3^3->6^3): " << max_err);
    CHECK(max_err < 1e-8);
}

// =============================================================================
// Test 3: Coarse → Fine mapping, quadratic field
// Coarse: 4×4×4 = 64, Fine: 8×8×8 = 512
// Quadratic field NOT exactly representable in trilinear Hex8 space
// → some approximation error expected, but should be small
// =============================================================================
TEST_CASE("3D MappingEngine: coarse-to-fine quadratic field accuracy", "[mapping_3d]") {
    // f(x,y,z) = x² + y² + z²  (quadratic, not in trilinear space)
    auto quad_f = [](double x, double y, double z) -> VectorXd {
        VectorXd v(1);
        v[0] = x*x + y*y + z*z;
        return v;
    };

    Mesh coarse = make_hex8_mesh(4, 4, 4, 1.0, 1.0, 1.0);
    Mesh fine   = make_hex8_mesh(8, 8, 8, 1.0, 1.0, 1.0);
    MatrixXd fd = make_field_data(coarse, quad_f, 1);

    MappingOptions3D opts;
    opts.n_threads = 1;
    MappingEngine3D engine(opts);

    MappingResult3D res = engine.map_integration_points(coarse, fine, fd, "Hex8");

    MatrixXd fd_fine = make_field_data(fine, quad_f, 1);

    double max_err = 0.0;
    double rms_err = 0.0;
    int n = res.values.rows();
    for (int i = 0; i < n; ++i) {
        double err = std::abs(res.values(i, 0) - fd_fine(i, 2));
        max_err = std::max(max_err, err);
        rms_err += err * err;
    }
    rms_err = std::sqrt(rms_err / n);
    INFO("Quadratic field (4^3->8^3): max_err=" << max_err << " rms_err=" << rms_err);
    // Quadratic field: trilinear approximation has O(h²) error
    // With h=1/4 on coarse mesh, expect error ~ 1/16 or so
    CHECK(max_err < 0.15);
    CHECK(rms_err < 0.05);
}

// =============================================================================
// Test 4: Multi-component field
// =============================================================================
TEST_CASE("3D MappingEngine: multi-component field", "[mapping_3d]") {
    // f(x,y,z) = [x+y, y+z, x+z]
    auto vec_f = [](double x, double y, double z) -> VectorXd {
        VectorXd v(3);
        v[0] = x + y;
        v[1] = y + z;
        v[2] = x + z;
        return v;
    };

    Mesh coarse = make_hex8_mesh(2, 2, 2, 1.0, 1.0, 1.0);
    Mesh fine   = make_hex8_mesh(4, 4, 4, 1.0, 1.0, 1.0);
    MatrixXd fd = make_field_data(coarse, vec_f, 3);

    MappingOptions3D opts;
    opts.n_threads = 1;
    MappingEngine3D engine(opts);

    MappingResult3D res = engine.map_integration_points(coarse, fine, fd, "Hex8");
    MatrixXd fd_fine = make_field_data(fine, vec_f, 3);

    REQUIRE(res.values.cols() == 3);

    double max_err = 0.0;
    for (int i = 0; i < res.values.rows(); ++i)
        for (int c = 0; c < 3; ++c)
            max_err = std::max(max_err, std::abs(res.values(i,c) - fd_fine(i, 2+c)));
    INFO("Multi-component max_err (2^3->4^3): " << max_err);
    CHECK(max_err < 1e-8);
}

// =============================================================================
// Test 5: Parallel == serial
// =============================================================================
TEST_CASE("3D MappingEngine: parallel == serial", "[mapping_3d]") {
    auto f = [](double x, double y, double z) -> VectorXd {
        VectorXd v(1); v[0] = 1.0 + x - y + 2.0*z; return v;
    };

    Mesh coarse = make_hex8_mesh(3, 3, 3, 1.0, 1.0, 1.0);
    Mesh fine   = make_hex8_mesh(5, 5, 5, 1.0, 1.0, 1.0);
    MatrixXd fd = make_field_data(coarse, f, 1);

    MappingOptions3D opts1; opts1.n_threads = 1;
    MappingOptions3D optsN; optsN.n_threads = 4;

    MappingEngine3D eng1(opts1), engN(optsN);
    MappingResult3D r1 = eng1.map_integration_points(coarse, fine, fd, "Hex8");
    MappingResult3D rN = engN.map_integration_points(coarse, fine, fd, "Hex8");

    REQUIRE(r1.values.rows() == rN.values.rows());
    double max_diff = (r1.values - rN.values).cwiseAbs().maxCoeff();
    CHECK(max_diff < 1e-12);
}

// =============================================================================
// Scale test helper: build mesh, map, measure time
// Returns: {time_ms, max_error, rms_error}
// =============================================================================
static std::tuple<double, double, double>
run_scale_test(int nx_coarse, int ny_coarse, int nz_coarse,
               int nx_fine,   int ny_fine,   int nz_fine,
               int n_threads = -1)
{
    // Test function: trilinear (exactly representable) so we can check accuracy
    auto f = [](double x, double y, double z) -> VectorXd {
        VectorXd v(1);
        v[0] = 1.0 + 2.0*x + 3.0*y + 4.0*z;
        return v;
    };

    Mesh coarse = make_hex8_mesh(nx_coarse, ny_coarse, nz_coarse);
    Mesh fine   = make_hex8_mesh(nx_fine,   ny_fine,   nz_fine);
    MatrixXd fd = make_field_data(coarse, f, 1);
    MatrixXd fd_fine = make_field_data(fine, f, 1);

    MappingOptions3D opts;
    opts.n_threads = n_threads;

    MappingEngine3D engine(opts);

    auto t0 = std::chrono::high_resolution_clock::now();
    MappingResult3D res = engine.map_integration_points(coarse, fine, fd, "Hex8");
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double max_err = 0.0, rms_err = 0.0;
    int n = res.values.rows();
    for (int i = 0; i < n; ++i) {
        double err = std::abs(res.values(i, 0) - fd_fine(i, 2));
        max_err = std::max(max_err, err);
        rms_err += err * err;
    }
    rms_err = std::sqrt(rms_err / n);
    return {elapsed_ms, max_err, rms_err};
}

// =============================================================================
// Test 6: Scale 1 000 fine elements (10³)
// Coarse: 5³ = 125, Fine: 10³ = 1000
// =============================================================================
TEST_CASE("3D Scale: 1K fine elements (5^3->10^3)", "[mapping_3d][scale]") {
    auto [ms, max_err, rms_err] = run_scale_test(5, 5, 5, 10, 10, 10);

    std::cout << "\n[Scale 1K]  coarse=5^3  fine=10^3"
              << "  time=" << ms << " ms"
              << "  max_err=" << max_err
              << "  rms_err=" << rms_err << "\n";

    CHECK(max_err < 1e-8);
}

// =============================================================================
// Test 7: Scale 10 000 fine elements (~21³ ≈ 9261, use 21³)
// Coarse: 10³ = 1000, Fine: 21³ = 9261
// =============================================================================
TEST_CASE("3D Scale: 10K fine elements (10^3->21^3)", "[mapping_3d][scale]") {
    auto [ms, max_err, rms_err] = run_scale_test(10, 10, 10, 21, 21, 21);

    std::cout << "\n[Scale 10K] coarse=10^3 fine=21^3"
              << "  time=" << ms << " ms"
              << "  max_err=" << max_err
              << "  rms_err=" << rms_err << "\n";

    CHECK(max_err < 1e-8);
}

// =============================================================================
// Test 8: Scale 100 000 fine elements (~46³ ≈ 97336, use 46³)
// Coarse: 22³ = 10648, Fine: 46³ = 97336
// =============================================================================
TEST_CASE("3D Scale: 100K fine elements (22^3->46^3)", "[mapping_3d][scale]") {
    auto [ms, max_err, rms_err] = run_scale_test(22, 22, 22, 46, 46, 46);

    std::cout << "\n[Scale 100K] coarse=22^3 fine=46^3"
              << "  time=" << ms << " ms"
              << "  max_err=" << max_err
              << "  rms_err=" << rms_err << "\n";

    CHECK(max_err < 1e-8);
}

// =============================================================================
// Exact engine tests (MappingEngine3D_Exact)
// =============================================================================

// =============================================================================
// Test 9: Exact engine — identity linear field (machine precision expected)
// =============================================================================
TEST_CASE("3D Exact: identity linear field", "[mapping_3d_exact]") {
    auto linear_f = [](double x, double y, double z) -> VectorXd {
        VectorXd v(1);
        v[0] = 1.0 + 2.0*x + 3.0*y + 4.0*z;
        return v;
    };

    Mesh mesh = make_hex8_mesh(4, 4, 4, 1.0, 1.0, 1.0);
    MatrixXd fd = make_field_data(mesh, linear_f, 1);

    MappingOptions3D_Exact opts;
    opts.n_threads = 1;
    MappingEngine3D_Exact engine(opts);

    auto res = engine.map_integration_points(mesh, mesh, fd, "Hex8");

    REQUIRE(res.values.rows() == fd.rows());
    REQUIRE(res.values.cols() == 1);

    double max_err = 0.0;
    for (int i = 0; i < res.values.rows(); ++i) {
        double expected = fd(i, 2);
        double got      = res.values(i, 0);
        max_err = std::max(max_err, std::abs(got - expected));
    }
    INFO("Exact engine max error (linear field, identity): " << max_err);
    CHECK(max_err < 1e-8);
}

// =============================================================================
// Test 10: Exact engine — coarse-to-fine linear field (machine precision)
// =============================================================================
TEST_CASE("3D Exact: coarse-to-fine linear field", "[mapping_3d_exact]") {
    auto linear_f = [](double x, double y, double z) -> VectorXd {
        VectorXd v(1);
        v[0] = 2.0 + 1.5*x - 0.5*y + 3.0*z;
        return v;
    };

    Mesh coarse = make_hex8_mesh(3, 3, 3, 1.0, 1.0, 1.0);
    Mesh fine   = make_hex8_mesh(6, 6, 6, 1.0, 1.0, 1.0);
    MatrixXd fd = make_field_data(coarse, linear_f, 1);

    MappingOptions3D_Exact opts;
    opts.n_threads = 1;
    MappingEngine3D_Exact engine(opts);

    auto res = engine.map_integration_points(coarse, fine, fd, "Hex8");

    MatrixXd fd_fine = make_field_data(fine, linear_f, 1);

    double max_err = 0.0;
    for (int i = 0; i < res.values.rows(); ++i) {
        double expected = fd_fine(i, 2);
        double got      = res.values(i, 0);
        max_err = std::max(max_err, std::abs(got - expected));
    }
    INFO("Exact engine max error (linear, 3^3->6^3): " << max_err);
    CHECK(max_err < 1e-8);
}

// =============================================================================
// Test 11: Exact engine — multi-component linear field
// =============================================================================
TEST_CASE("3D Exact: multi-component linear field", "[mapping_3d_exact]") {
    auto vec_f = [](double x, double y, double z) -> VectorXd {
        VectorXd v(3);
        v[0] = x + y;
        v[1] = y + z;
        v[2] = x + z;
        return v;
    };

    Mesh coarse = make_hex8_mesh(2, 2, 2, 1.0, 1.0, 1.0);
    Mesh fine   = make_hex8_mesh(4, 4, 4, 1.0, 1.0, 1.0);
    MatrixXd fd = make_field_data(coarse, vec_f, 3);

    MappingOptions3D_Exact opts;
    opts.n_threads = 1;
    MappingEngine3D_Exact engine(opts);

    auto res = engine.map_integration_points(coarse, fine, fd, "Hex8");
    MatrixXd fd_fine = make_field_data(fine, vec_f, 3);

    REQUIRE(res.values.cols() == 3);

    double max_err = 0.0;
    for (int i = 0; i < res.values.rows(); ++i)
        for (int c = 0; c < 3; ++c)
            max_err = std::max(max_err, std::abs(res.values(i,c) - fd_fine(i, 2+c)));
    INFO("Exact engine multi-component max_err (2^3->4^3): " << max_err);
    CHECK(max_err < 1e-8);
}

// =============================================================================
// Test 12: Exact vs approximate — non-conforming meshes, linear field
// =============================================================================
TEST_CASE("3D Exact vs Approx: linear field (3^3->5^3)", "[mapping_3d_exact]") {
    auto linear_f = [](double x, double y, double z) -> VectorXd {
        VectorXd v(1);
        v[0] = 1.0 + x - 2.0*y + 3.0*z;
        return v;
    };

    Mesh coarse = make_hex8_mesh(3, 3, 3, 1.0, 1.0, 1.0);
    Mesh fine   = make_hex8_mesh(5, 5, 5, 1.0, 1.0, 1.0);
    MatrixXd fd      = make_field_data(coarse, linear_f, 1);
    MatrixXd fd_fine = make_field_data(fine,   linear_f, 1);

    MappingOptions3D       opts_approx; opts_approx.n_threads = 1;
    MappingOptions3D_Exact opts_exact;  opts_exact.n_threads  = 1;

    MappingEngine3D       eng_approx(opts_approx);
    MappingEngine3D_Exact eng_exact(opts_exact);

    MappingResult3D r_approx = eng_approx.map_integration_points(coarse, fine, fd, "Hex8");
    auto            r_exact  = eng_exact .map_integration_points(coarse, fine, fd, "Hex8");

    double max_err_approx = 0.0, max_err_exact = 0.0;
    for (int i = 0; i < r_approx.values.rows(); ++i) {
        double expected = fd_fine(i, 2);
        max_err_approx = std::max(max_err_approx, std::abs(r_approx.values(i,0) - expected));
        max_err_exact  = std::max(max_err_exact,  std::abs(r_exact .values(i,0) - expected));
    }
    std::cout << "\n[Exact vs Approx, linear 3^3->5^3]"
              << "  approx max_err=" << max_err_approx
              << "  exact  max_err=" << max_err_exact << "\n";

    CHECK(max_err_exact  < 1e-8);
    CHECK(max_err_approx < 1e-8);
}

// =============================================================================
// Test 13: Exact engine scale — 1K fine elements
// =============================================================================
TEST_CASE("3D Exact Scale: 1K fine elements (5^3->10^3)", "[mapping_3d_exact][scale]") {
    auto f = [](double x, double y, double z) -> VectorXd {
        VectorXd v(1); v[0] = 1.0 + 2.0*x + 3.0*y + 4.0*z; return v;
    };

    Mesh coarse  = make_hex8_mesh(5, 5, 5);
    Mesh fine    = make_hex8_mesh(10, 10, 10);
    MatrixXd fd      = make_field_data(coarse, f, 1);
    MatrixXd fd_fine = make_field_data(fine,   f, 1);

    MappingOptions3D_Exact opts; opts.n_threads = -1;
    MappingEngine3D_Exact engine(opts);

    auto t0  = std::chrono::high_resolution_clock::now();
    auto res = engine.map_integration_points(coarse, fine, fd, "Hex8");
    auto t1  = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double max_err = 0.0;
    for (int i = 0; i < res.values.rows(); ++i)
        max_err = std::max(max_err, std::abs(res.values(i,0) - fd_fine(i,2)));

    std::cout << "\n[Exact Scale 1K]  coarse=5^3  fine=10^3"
              << "  time=" << ms << " ms"
              << "  max_err=" << max_err << "\n";

    CHECK(max_err < 1e-8);
}

// =============================================================================
// Test 14: Exact engine scale — 10K fine elements (10³→21³)
// =============================================================================
TEST_CASE("3D Exact Scale: 10K fine elements (10^3->21^3)", "[mapping_3d_exact][scale]") {
    auto f = [](double x, double y, double z) -> VectorXd {
        VectorXd v(1); v[0] = 1.0 + 2.0*x + 3.0*y + 4.0*z; return v;
    };

    Mesh coarse  = make_hex8_mesh(10, 10, 10);
    Mesh fine    = make_hex8_mesh(21, 21, 21);
    MatrixXd fd      = make_field_data(coarse, f, 1);
    MatrixXd fd_fine = make_field_data(fine,   f, 1);

    MappingOptions3D_Exact opts; opts.n_threads = -1;
    MappingEngine3D_Exact engine(opts);

    auto t0  = std::chrono::high_resolution_clock::now();
    auto res = engine.map_integration_points(coarse, fine, fd, "Hex8");
    auto t1  = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double max_err = 0.0;
    for (int i = 0; i < res.values.rows(); ++i)
        max_err = std::max(max_err, std::abs(res.values(i,0) - fd_fine(i,2)));

    std::cout << "\n[Exact Scale 10K] coarse=10^3 fine=21^3"
              << "  time=" << ms << " ms"
              << "  max_err=" << max_err << "\n";

    CHECK(max_err < 1e-8);
}

// =============================================================================
// Test 15: Exact engine scale — 100K fine elements (22³→46³)
// =============================================================================
TEST_CASE("3D Exact Scale: 100K fine elements (22^3->46^3)", "[mapping_3d_exact][scale]") {
    auto f = [](double x, double y, double z) -> VectorXd {
        VectorXd v(1); v[0] = 1.0 + 2.0*x + 3.0*y + 4.0*z; return v;
    };

    Mesh coarse  = make_hex8_mesh(22, 22, 22);
    Mesh fine    = make_hex8_mesh(46, 46, 46);
    MatrixXd fd      = make_field_data(coarse, f, 1);
    MatrixXd fd_fine = make_field_data(fine,   f, 1);

    MappingOptions3D_Exact opts; opts.n_threads = -1;
    MappingEngine3D_Exact engine(opts);

    auto t0  = std::chrono::high_resolution_clock::now();
    auto res = engine.map_integration_points(coarse, fine, fd, "Hex8");
    auto t1  = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double max_err = 0.0;
    for (int i = 0; i < res.values.rows(); ++i)
        max_err = std::max(max_err, std::abs(res.values(i,0) - fd_fine(i,2)));

    std::cout << "\n[Exact Scale 100K] coarse=22^3 fine=46^3"
              << "  time=" << ms << " ms"
              << "  max_err=" << max_err << "\n";

    CHECK(max_err < 1e-8);
}

// =============================================================================
// Tet4 element library tests
// =============================================================================

// Unit tetrahedron: N0(0,0,0), N1(1,0,0), N2(0,1,0), N3(0,0,1)
static MatrixXd make_unit_tet4_nodes() {
    MatrixXd nodes(4, 3);
    nodes << 0.0, 0.0, 0.0,
             1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0;
    return nodes;
}

TEST_CASE("Tet4: partition of unity at Gauss points", "[tet4][element_library_3d]") {
    const auto& et = ElementLibrary3D::instance().get("Tet4");
    REQUIRE(et.n_integration_points == 4);

    for (int q = 0; q < et.n_integration_points; ++q) {
        double xi   = et.gauss_pts_natural[q][0];
        double eta  = et.gauss_pts_natural[q][1];
        double zeta = et.gauss_pts_natural[q][2];
        VectorXd N  = et.shape_functions(xi, eta, zeta);
        CHECK_THAT(N.sum(), Catch::Matchers::WithinAbs(1.0, 1e-14));
    }
}

TEST_CASE("Tet4: Gauss weights sum to 1/6 (reference tet volume)", "[tet4][element_library_3d]") {
    const auto& et = ElementLibrary3D::instance().get("Tet4");
    double total = 0.0;
    for (double w : et.gauss_weights) total += w;
    CHECK_THAT(total, Catch::Matchers::WithinAbs(1.0 / 6.0, 1e-14));
}

TEST_CASE("Tet4: Gauss points map inside unit tet", "[tet4][element_library_3d]") {
    MatrixXd nodes = make_unit_tet4_nodes();
    auto gpts = ElementLibrary3D::instance().compute_gauss_points_global("Tet4", nodes);
    REQUIRE(static_cast<int>(gpts.size()) == 4);

    for (const auto& p : gpts) {
        CHECK(p[0] > 0.0);
        CHECK(p[1] > 0.0);
        CHECK(p[2] > 0.0);
        CHECK(p[0] + p[1] + p[2] < 1.0);
    }
}

TEST_CASE("Tet4: Jacobian determinant = 1 for unit tet", "[tet4][element_library_3d]") {
    // Unit tet has identity Jacobian, so |J| = 1.
    MatrixXd nodes = make_unit_tet4_nodes();
    double jdet = ElementLibrary3D::instance().jacobian_det("Tet4", nodes, 0.25, 0.25, 0.25);
    CHECK_THAT(jdet, Catch::Matchers::WithinAbs(1.0, 1e-14));
}

TEST_CASE("Tet4: Lagrange delta property via BasisBuilder3D", "[tet4][basis_builder_3d]") {
    MatrixXd nodes = make_unit_tet4_nodes();
    auto gpts = ElementLibrary3D::instance().compute_gauss_points_global("Tet4", nodes);
    int N = static_cast<int>(gpts.size());

    BasisBuilder3D bb;
    BasisMatrix basis = bb.build(gpts);
    MonomialBasis3D mono = get_tensor_basis_3d(N);  // {1, x, y, z}

    // builder shifts by gpts.back() — mirror that here
    Point3D origin = gpts.back();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Point3D p_sh = gpts[j] - origin;
            VectorXd m = mono.evaluate(p_sh[0], p_sh[1], p_sh[2]);
            double val = basis.row(i).dot(m);
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK_THAT(val, Catch::Matchers::WithinAbs(expected, 1e-10));
        }
    }
}
