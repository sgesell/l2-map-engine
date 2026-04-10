#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "l2map/mapping_engine.hpp"
#include "l2map/element_library.hpp"
#include "l2map/mesh.hpp"
#include "l2map/io.hpp"
#include <cmath>
#include <string>
#include <fstream>

using namespace l2map;

// ---------------------------------------------------------------------------
// Helpers: build a small regular 2×2 grid of Quad8 elements
// Grid covers [0,2]×[0,2] with 2 elements per side = 4 elements total.
// Each element is a 1×1 Quad8 (corners + midsides).
// ---------------------------------------------------------------------------

static std::pair<Mesh, MatrixXd> make_test_mesh_and_field(
    int nx = 2, double scale = 1.0, double field_offset = 0.0)
{
    // Build a regular nx×nx grid of Quad8 elements.
    // Node numbering: column-major, 3 nodes per element dimension (1 corner + 1 midside)
    // Total nodes per direction: 2*nx + 1
    int npts = 2 * nx + 1;
    double h = scale / nx;  // element size

    // Generate nodes
    std::vector<Node> nodes;
    NodeID nid = 0;
    for (int j = 0; j < npts; ++j) {
        for (int i = 0; i < npts; ++i) {
            Node nd;
            nd.id = nid++;
            nd.x  = i * h * 0.5;
            nd.y  = j * h * 0.5;
            nodes.push_back(nd);
        }
    }

    // Generate elements: each Quad8 element in a nx×nx grid
    std::vector<Element> elements;
    ElemID eid = 0;
    for (int ej = 0; ej < nx; ++ej) {
        for (int ei = 0; ei < nx; ++ei) {
            // Bottom-left corner node of this element (in 2h×2h node grid)
            int ci = 2 * ei;
            int cj = 2 * ej;
            auto nidx = [&](int di, int dj) -> NodeID {
                return static_cast<NodeID>((cj + dj) * npts + (ci + di));
            };
            Element e;
            e.id = eid++;
            e.type_name = "Quad8";
            // Corner nodes: BL, BR, TR, TL
            e.node_ids = {nidx(0,0), nidx(2,0), nidx(2,2), nidx(0,2),
                          // Midside nodes: B, R, T, L
                          nidx(1,0), nidx(2,1), nidx(1,2), nidx(0,1)};
            elements.push_back(e);
        }
    }
    Mesh mesh(nodes, elements, "Quad8");

    // Build field data: a linear field u(x, y) = 1 + 2x + 3y + field_offset
    // 9 integration points per element
    const auto& elem_ids = mesh.element_ids();
    int n_elem = static_cast<int>(elem_ids.size());
    MatrixXd fd(n_elem * 9, 3);  // [elem_id(1-idx), ipt(1-idx), u_value]

    int row = 0;
    for (ElemID eid2 : elem_ids) {
        MatrixXd coords = mesh.element_node_coords(eid2);
        auto gauss_pts = ElementLibrary::instance()
            .compute_gauss_points_global("Quad8", coords);
        for (int ipt = 0; ipt < 9; ++ipt) {
            double x = gauss_pts[ipt][0];
            double y = gauss_pts[ipt][1];
            double u = 1.0 + 2.0 * x + 3.0 * y + field_offset;
            fd(row, 0) = static_cast<double>(eid2 + 1);  // 1-indexed
            fd(row, 1) = static_cast<double>(ipt + 1);
            fd(row, 2) = u;
            ++row;
        }
    }
    return {mesh, fd};
}

// ---------------------------------------------------------------------------
// Test 1: Fixed mesh identity — map from mesh to itself, result ≈ input
// ---------------------------------------------------------------------------
TEST_CASE("MappingEngine: fixed mesh identity", "[mapping_engine]") {
    auto [mesh, field_data] = make_test_mesh_and_field(2, 2.0);

    MappingOptions opts;
    opts.verbose   = false;
    opts.n_threads = 1;
    MappingEngine engine(opts);

    MappingResult res = engine.map_integration_points(mesh, mesh, field_data);

    int n_rows = static_cast<int>(res.values.rows());
    REQUIRE(n_rows == static_cast<int>(field_data.rows()));

    // Compare mapped values to original field data
    double max_err = 0.0;
    for (int i = 0; i < n_rows; ++i) {
        double original = field_data(i, 2);
        double mapped   = res.values(i, 0);
        max_err = std::max(max_err, std::abs(mapped - original));
    }
    // Should be exact for a polynomial field within the polynomial space
    CHECK(max_err < 1e-6);
}

// ---------------------------------------------------------------------------
// Test 2: Parallelism correctness — 1 thread vs N threads give same result
// ---------------------------------------------------------------------------
TEST_CASE("MappingEngine: parallel = serial result", "[mapping_engine]") {
    auto [old_mesh, field_data] = make_test_mesh_and_field(2, 2.0);
    auto [new_mesh, fd_unused1] = make_test_mesh_and_field(2, 2.0, 0.01);
    (void)fd_unused1;

    MappingOptions opts1;
    opts1.n_threads = 1;
    MappingEngine eng1(opts1);
    MappingResult r1 = eng1.map_integration_points(old_mesh, new_mesh, field_data);

    MappingOptions optsN;
    optsN.n_threads = 4;
    MappingEngine engN(optsN);
    MappingResult rN = engN.map_integration_points(old_mesh, new_mesh, field_data);

    REQUIRE(r1.values.rows() == rN.values.rows());
    REQUIRE(r1.values.cols() == rN.values.cols());

    double max_diff = (r1.values - rN.values).cwiseAbs().maxCoeff();
    CHECK(max_diff < 1e-12);
}

// ---------------------------------------------------------------------------
// Test 3: Positive enforcement — result contains no negative values
// ---------------------------------------------------------------------------
TEST_CASE("MappingEngine: enforce_positive clips to zero", "[mapping_engine]") {
    auto [mesh, field_data] = make_test_mesh_and_field(2, 2.0);

    // Set all field values to a mix of positive and negative
    for (int i = 0; i < field_data.rows(); ++i) {
        double x = field_data(i, 2);
        field_data(i, 2) = x - 3.0;  // subtract 3 to introduce negatives
    }

    MappingOptions opts;
    opts.enforce_positive = true;
    opts.n_threads = 1;
    MappingEngine engine(opts);
    MappingResult res = engine.map_integration_points(mesh, mesh, field_data);

    for (int i = 0; i < res.values.rows(); ++i)
        CHECK(res.values(i, 0) >= 0.0);
}

// ---------------------------------------------------------------------------
// Test 4: Full mesh integration test (if reference data exists)
// ---------------------------------------------------------------------------
TEST_CASE("MappingEngine: full mesh vs reference data", "[mapping_engine][reference]") {
    std::string old_file = std::string(TEST_DATA_DIR) + "/integration_test_mesh_old.txt";
    std::string new_file = std::string(TEST_DATA_DIR) + "/integration_test_mesh_new.txt";
    std::string fld_file = std::string(TEST_DATA_DIR) + "/integration_test_expected_output.txt";

    std::ifstream f1(old_file), f2(new_file), f3(fld_file);
    if (!f1.is_open() || !f2.is_open() || !f3.is_open()) {
        WARN("Reference data not found, skipping full integration test");
        return;
    }
    f1.close(); f2.close(); f3.close();

    // Load meshes and field data
    auto old_nodes = io::read_nodes(old_file);
    // (In the actual integration test, separate node/element files would be used.
    //  This test case serves as a placeholder — populate reference_data/ to activate it.)
    WARN("Integration test reference files found but mesh format requires node+element files."
         " Populate tests/reference_data/ with the correct files to enable full comparison.");
}

// ---------------------------------------------------------------------------
// Test 5: MappingResult has correct dimensions
// ---------------------------------------------------------------------------
TEST_CASE("MappingEngine: output dimensions are correct", "[mapping_engine]") {
    int nx = 2;
    auto [old_mesh, field_data] = make_test_mesh_and_field(nx, 2.0);
    auto [new_mesh, fd_unused2] = make_test_mesh_and_field(nx, 2.0);
    (void)fd_unused2;

    MappingOptions opts;
    opts.n_threads = 1;
    MappingEngine engine(opts);
    MappingResult res = engine.map_integration_points(old_mesh, new_mesh, field_data);

    int n_elems = nx * nx;     // 4
    int n_ipts  = 9;
    int n_comp  = 1;
    CHECK(res.values.rows() == n_elems * n_ipts);
    CHECK(res.values.cols() == n_comp);
    CHECK(static_cast<int>(res.ipoint_coords.size()) == n_elems * n_ipts);
}
