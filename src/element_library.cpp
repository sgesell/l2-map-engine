#include "l2map/element_library.hpp"
#include <cmath>
#include <stdexcept>

namespace l2map {

ElementLibrary& ElementLibrary::instance() {
    static ElementLibrary inst;
    return inst;
}

ElementLibrary::ElementLibrary() {
    register_builtins_();
}

void ElementLibrary::register_type(const ElementType& et) {
    types_[et.name] = et;
}

const ElementType& ElementLibrary::get(const std::string& name) const {
    auto it = types_.find(name);
    if (it == types_.end())
        throw std::runtime_error("ElementLibrary: unknown element type '" + name + "'");
    return it->second;
}

std::vector<Point2D> ElementLibrary::compute_gauss_points_global(
    const std::string& type_name,
    const MatrixXd& nodes_global) const
{
    const ElementType& et = get(type_name);
    std::vector<Point2D> gpts;
    gpts.reserve(et.n_integration_points);
    for (const auto& nat : et.gauss_pts_natural) {
        VectorXd N = et.shape_functions(nat[0], nat[1]);
        // x = N^T * x_nodes,  y = N^T * y_nodes
        double x = N.dot(nodes_global.col(0));
        double y = N.dot(nodes_global.col(1));
        gpts.push_back(Point2D(x, y));
    }
    return gpts;
}

Polygon2D ElementLibrary::element_polygon(
    const std::string& type_name,
    const MatrixXd& nodes_global) const
{
    const ElementType& et = get(type_name);
    Polygon2D poly;
    poly.reserve(et.polygon_vertex_order.size());
    for (int idx : et.polygon_vertex_order) {
        poly.push_back(Point2D(nodes_global(idx, 0), nodes_global(idx, 1)));
    }
    return poly;
}

// ---------------------------------------------------------------------------
// Quad8 — 8-node serendipity quadrilateral, 9-point (3×3) Gauss rule
// ---------------------------------------------------------------------------
static VectorXd quad8_shape(double xi, double eta) {
    VectorXd N(8);
    N(0) = -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta);
    N(1) = -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta);
    N(2) = -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta);
    N(3) = -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta);
    N(4) =  0.5  * (1 - xi * xi) * (1 - eta);
    N(5) =  0.5  * (1 + xi)      * (1 - eta * eta);
    N(6) =  0.5  * (1 - xi * xi) * (1 + eta);
    N(7) =  0.5  * (1 - xi)      * (1 - eta * eta);
    return N;
}

// ---------------------------------------------------------------------------
// Quad4 — 4-node bilinear quadrilateral, 4-point (2×2) Gauss rule
// ---------------------------------------------------------------------------
static VectorXd quad4_shape(double xi, double eta) {
    VectorXd N(4);
    N(0) = 0.25 * (1 - xi) * (1 - eta);
    N(1) = 0.25 * (1 + xi) * (1 - eta);
    N(2) = 0.25 * (1 + xi) * (1 + eta);
    N(3) = 0.25 * (1 - xi) * (1 + eta);
    return N;
}

// ---------------------------------------------------------------------------
// Quad9 — 9-node Lagrangian quadrilateral, 9-point Gauss rule
// ---------------------------------------------------------------------------
static VectorXd quad9_shape(double xi, double eta) {
    VectorXd N(9);
    // Corner nodes
    N(0) = 0.25 * xi * (xi - 1) * eta * (eta - 1);
    N(1) = 0.25 * xi * (xi + 1) * eta * (eta - 1);
    N(2) = 0.25 * xi * (xi + 1) * eta * (eta + 1);
    N(3) = 0.25 * xi * (xi - 1) * eta * (eta + 1);
    // Edge midside nodes
    N(4) = 0.5 * (1 - xi * xi) * eta * (eta - 1);
    N(5) = 0.5 * xi * (xi + 1) * (1 - eta * eta);
    N(6) = 0.5 * (1 - xi * xi) * eta * (eta + 1);
    N(7) = 0.5 * xi * (xi - 1) * (1 - eta * eta);
    // Center node
    N(8) = (1 - xi * xi) * (1 - eta * eta);
    return N;
}

// ---------------------------------------------------------------------------
// Tri6 — 6-node quadratic triangle, 6-point Dunavant rule (exact degree 4)
//
// Node ordering (L1=xi, L2=eta, L3=1-xi-eta):
//   N0: corner (0, 0)    N1: corner (1, 0)    N2: corner (0, 1)
//   N3: midside N0-N1 (0.5, 0)
//   N4: midside N1-N2 (0.5, 0.5)
//   N5: midside N2-N0 (0,   0.5)
// ---------------------------------------------------------------------------
static VectorXd tri6_shape(double xi, double eta) {
    const double L1 = xi;
    const double L2 = eta;
    const double L3 = 1.0 - xi - eta;
    VectorXd N(6);
    N(0) = L3 * (2.0*L3 - 1.0);
    N(1) = L1 * (2.0*L1 - 1.0);
    N(2) = L2 * (2.0*L2 - 1.0);
    N(3) = 4.0 * L1 * L3;
    N(4) = 4.0 * L1 * L2;
    N(5) = 4.0 * L2 * L3;
    return N;
}

void ElementLibrary::register_builtins_() {
    // --- Quad8 ---
    {
        ElementType et;
        et.name = "Quad8";
        et.dim  = Dimension::D2;
        et.n_nodes = 8;
        et.n_integration_points = 9;
        et.poly_degree = 2;
        et.n_monomials = 9;  // full degree-2 basis (matches Python reference)

        // 3×3 Gauss-Legendre rule, Abaqus column-major ordering
        const double gp = std::sqrt(0.6);   // ≈ 0.7745966...
        // column-major: eta varies first (outer), xi inner? No: Abaqus ordering is:
        // pt0=(xi=-gp, eta=-gp), pt1=(xi=0, eta=-gp), pt2=(xi=+gp, eta=-gp)
        // pt3=(xi=-gp, eta= 0 ), pt4=(xi=0, eta= 0 ), pt5=(xi=+gp, eta= 0 )
        // pt6=(xi=-gp, eta=+gp), pt7=(xi=0, eta=+gp), pt8=(xi=+gp, eta=+gp)
        const double xi_vals[3]  = {-gp, 0.0,  gp};
        const double eta_vals[3] = {-gp, 0.0,  gp};
        const double w_edge   = 5.0 / 9.0;
        const double w_center = 8.0 / 9.0;
        const double wxi[3]  = {w_edge, w_center, w_edge};
        const double weta[3] = {w_edge, w_center, w_edge};

        for (int ieta = 0; ieta < 3; ++ieta) {
            for (int ixi = 0; ixi < 3; ++ixi) {
                et.gauss_pts_natural.push_back(Point2D(xi_vals[ixi], eta_vals[ieta]));
                et.gauss_weights.push_back(wxi[ixi] * weta[ieta]);
            }
        }

        // Node natural coordinates (corners then midside, consistent with shape fn ordering)
        et.node_pts_natural = {
            {-1, -1}, { 1, -1}, { 1,  1}, {-1,  1},  // corners N0-N3
            { 0, -1}, { 1,  0}, { 0,  1}, {-1,  0}   // midsides N4-N7
        };

        // CCW polygon traversal through all 8 nodes
        et.polygon_vertex_order = {0, 4, 1, 5, 2, 6, 3, 7};

        et.shape_functions = quad8_shape;
        register_type(et);
    }

    // --- Quad4 ---
    {
        ElementType et;
        et.name = "Quad4";
        et.dim  = Dimension::D2;
        et.n_nodes = 4;
        et.n_integration_points = 4;
        et.poly_degree = 1;
        et.n_monomials = 4;

        const double gp = 1.0 / std::sqrt(3.0);
        et.gauss_pts_natural = {
            {-gp, -gp}, { gp, -gp},
            { gp,  gp}, {-gp,  gp}
        };
        et.gauss_weights = {1.0, 1.0, 1.0, 1.0};

        et.node_pts_natural = {
            {-1, -1}, {1, -1}, {1, 1}, {-1, 1}
        };
        et.polygon_vertex_order = {0, 1, 2, 3};
        et.shape_functions = quad4_shape;
        register_type(et);
    }

    // --- Quad9 ---
    {
        ElementType et;
        et.name = "Quad9";
        et.dim  = Dimension::D2;
        et.n_nodes = 9;
        et.n_integration_points = 9;
        et.poly_degree = 2;
        et.n_monomials = 9;

        const double gp = std::sqrt(0.6);
        const double xi_vals[3]  = {-gp, 0.0,  gp};
        const double eta_vals[3] = {-gp, 0.0,  gp};
        const double w_edge   = 5.0 / 9.0;
        const double w_center = 8.0 / 9.0;
        const double wxi[3]  = {w_edge, w_center, w_edge};
        const double weta[3] = {w_edge, w_center, w_edge};

        for (int ieta = 0; ieta < 3; ++ieta) {
            for (int ixi = 0; ixi < 3; ++ixi) {
                et.gauss_pts_natural.push_back(Point2D(xi_vals[ixi], eta_vals[ieta]));
                et.gauss_weights.push_back(wxi[ixi] * weta[ieta]);
            }
        }

        et.node_pts_natural = {
            {-1,-1},{1,-1},{1,1},{-1,1},  // corners
            {0,-1},{1,0},{0,1},{-1,0},    // midsides
            {0,0}                          // center
        };
        // Center node (8) not part of the polygon boundary
        et.polygon_vertex_order = {0, 4, 1, 5, 2, 6, 3, 7};
        et.shape_functions = quad9_shape;
        register_type(et);
    }

    // --- Tri6 ---
    {
        ElementType et;
        et.name = "Tri6";
        et.dim  = Dimension::D2;
        et.n_nodes = 6;
        et.n_integration_points = 6;
        et.poly_degree = 2;
        et.n_monomials = 6;  // degree-2 triangle basis: {1, x, y, x², xy, y²}

        // Dunavant 6-point rule, exact for degree 4.
        // Two families of 3 symmetrically placed points.
        // Reference triangle: (0,0)-(1,0)-(0,1), area = 0.5.
        // Weights already include the 1/2 area factor.
        const double a1 = 0.44594849091597, b1 = 0.10810301816807;
        const double a2 = 0.09157621350977, b2 = 0.81684757298046;
        const double w1 = 0.11169079483901;  // 0.22338158967801 / 2
        const double w2 = 0.05497587182766;  // 0.10995174365532 / 2

        et.gauss_pts_natural = {
            {a1, a1}, {b1, a1}, {a1, b1},   // family 1
            {a2, a2}, {b2, a2}, {a2, b2},   // family 2
        };
        et.gauss_weights = { w1, w1, w1, w2, w2, w2 };

        et.node_pts_natural = {
            {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0},   // corners N0-N2
            {0.5, 0.0}, {0.5, 0.5}, {0.0, 0.5},   // midsides N3-N5
        };

        // CCW: corner, midside, corner, midside, corner, midside
        et.polygon_vertex_order = {0, 3, 1, 4, 2, 5};

        et.shape_functions = tri6_shape;
        register_type(et);
    }
}

} // namespace l2map
