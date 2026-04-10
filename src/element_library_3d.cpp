#include "l2map/element_library_3d.hpp"
#include <cmath>
#include <stdexcept>

namespace l2map {

ElementLibrary3D& ElementLibrary3D::instance() {
    static ElementLibrary3D inst;
    return inst;
}

ElementLibrary3D::ElementLibrary3D() {
    register_builtins_();
}

void ElementLibrary3D::register_type(const ElementType3D& et) {
    types_[et.name] = et;
}

const ElementType3D& ElementLibrary3D::get(const std::string& name) const {
    auto it = types_.find(name);
    if (it == types_.end())
        throw std::runtime_error("ElementLibrary3D: unknown element type '" + name + "'");
    return it->second;
}

std::vector<Point3D> ElementLibrary3D::compute_gauss_points_global(
    const std::string& type_name,
    const MatrixXd& node_coords) const
{
    const ElementType3D& et = get(type_name);
    std::vector<Point3D> pts;
    pts.reserve(et.n_integration_points);
    for (const auto& g : et.gauss_pts_natural) {
        VectorXd N = et.shape_functions(g[0], g[1], g[2]);
        // x = N^T * node_coords
        Point3D p = node_coords.transpose() * N;
        pts.push_back(p);
    }
    return pts;
}

void ElementLibrary3D::compute_quad_points_global(
    const std::string& type_name,
    const MatrixXd& node_coords,
    std::vector<Point3D>& phys_pts_out,
    std::vector<double>&  phys_weights_out) const
{
    const ElementType3D& et = get(type_name);
    int nq = et.n_quad_points;
    phys_pts_out.resize(nq);
    phys_weights_out.resize(nq);

    for (int q = 0; q < nq; ++q) {
        double xi  = et.quad_pts_natural[q][0];
        double eta = et.quad_pts_natural[q][1];
        double zet = et.quad_pts_natural[q][2];

        VectorXd N = et.shape_functions(xi, eta, zet);
        phys_pts_out[q] = node_coords.transpose() * N;

        double jdet = jacobian_det(type_name, node_coords, xi, eta, zet);
        phys_weights_out[q] = et.quad_weights[q] * std::abs(jdet);
    }
}

double ElementLibrary3D::jacobian_det(
    const std::string& type_name,
    const MatrixXd& node_coords,
    double xi, double eta, double zeta) const
{
    const ElementType3D& et = get(type_name);
    // dN/d(xi,eta,zeta): (n_nodes x 3)
    MatrixXd dN = et.shape_fn_gradients(xi, eta, zeta);
    // J = dN^T * node_coords: (3x3)   [column = physical gradient direction]
    // Actually J[i,j] = sum_k dN_k/dxi_i * x_k^j
    // J = dN.transpose() * node_coords  is (3 x 3)
    MatrixXd J = dN.transpose() * node_coords;  // (3 x 3)
    return J.determinant();
}

// ---------------------------------------------------------------------------
// Hex8 — 8-node trilinear hexahedron
//
// Node numbering (Abaqus / standard):
//   Bottom face (z=-1):  N0(-1,-1,-1), N1(+1,-1,-1), N2(+1,+1,-1), N3(-1,+1,-1)
//   Top face    (z=+1):  N4(-1,-1,+1), N5(+1,-1,+1), N6(+1,+1,+1), N7(-1,+1,+1)
//
// Shape function: N_i = (1 ± xi)(1 ± eta)(1 ± zeta) / 8
// ---------------------------------------------------------------------------

static VectorXd hex8_shape(double xi, double eta, double zeta) {
    VectorXd N(8);
    N(0) = 0.125 * (1 - xi) * (1 - eta) * (1 - zeta);
    N(1) = 0.125 * (1 + xi) * (1 - eta) * (1 - zeta);
    N(2) = 0.125 * (1 + xi) * (1 + eta) * (1 - zeta);
    N(3) = 0.125 * (1 - xi) * (1 + eta) * (1 - zeta);
    N(4) = 0.125 * (1 - xi) * (1 - eta) * (1 + zeta);
    N(5) = 0.125 * (1 + xi) * (1 - eta) * (1 + zeta);
    N(6) = 0.125 * (1 + xi) * (1 + eta) * (1 + zeta);
    N(7) = 0.125 * (1 - xi) * (1 + eta) * (1 + zeta);
    return N;
}

// Returns (8 x 3) matrix: row i = [dN_i/dxi, dN_i/deta, dN_i/dzeta]
static MatrixXd hex8_grad(double xi, double eta, double zeta) {
    MatrixXd dN(8, 3);
    // dN/dxi
    dN(0, 0) = -0.125 * (1 - eta) * (1 - zeta);
    dN(1, 0) =  0.125 * (1 - eta) * (1 - zeta);
    dN(2, 0) =  0.125 * (1 + eta) * (1 - zeta);
    dN(3, 0) = -0.125 * (1 + eta) * (1 - zeta);
    dN(4, 0) = -0.125 * (1 - eta) * (1 + zeta);
    dN(5, 0) =  0.125 * (1 - eta) * (1 + zeta);
    dN(6, 0) =  0.125 * (1 + eta) * (1 + zeta);
    dN(7, 0) = -0.125 * (1 + eta) * (1 + zeta);
    // dN/deta
    dN(0, 1) = -0.125 * (1 - xi) * (1 - zeta);
    dN(1, 1) = -0.125 * (1 + xi) * (1 - zeta);
    dN(2, 1) =  0.125 * (1 + xi) * (1 - zeta);
    dN(3, 1) =  0.125 * (1 - xi) * (1 - zeta);
    dN(4, 1) = -0.125 * (1 - xi) * (1 + zeta);
    dN(5, 1) = -0.125 * (1 + xi) * (1 + zeta);
    dN(6, 1) =  0.125 * (1 + xi) * (1 + zeta);
    dN(7, 1) =  0.125 * (1 - xi) * (1 + zeta);
    // dN/dzeta
    dN(0, 2) = -0.125 * (1 - xi) * (1 - eta);
    dN(1, 2) = -0.125 * (1 + xi) * (1 - eta);
    dN(2, 2) = -0.125 * (1 + xi) * (1 + eta);
    dN(3, 2) = -0.125 * (1 - xi) * (1 + eta);
    dN(4, 2) =  0.125 * (1 - xi) * (1 - eta);
    dN(5, 2) =  0.125 * (1 + xi) * (1 - eta);
    dN(6, 2) =  0.125 * (1 + xi) * (1 + eta);
    dN(7, 2) =  0.125 * (1 - xi) * (1 + eta);
    return dN;
}

// ---------------------------------------------------------------------------
// Tet4 — 4-node linear tetrahedron
//
// Node ordering (L0=1-xi-eta-zeta, L1=xi, L2=eta, L3=zeta):
//   N0: (0,0,0)   N1: (1,0,0)   N2: (0,1,0)   N3: (0,0,1)
//
// Shape functions:
//   N0 = 1-xi-eta-zeta   N1 = xi   N2 = eta   N3 = zeta
// ---------------------------------------------------------------------------

static VectorXd tet4_shape(double xi, double eta, double zeta) {
    VectorXd N(4);
    N(0) = 1.0 - xi - eta - zeta;
    N(1) = xi;
    N(2) = eta;
    N(3) = zeta;
    return N;
}

// Returns (4 x 3): row i = [dN_i/dxi, dN_i/deta, dN_i/dzeta]
// Gradients are constant for linear element.
static MatrixXd tet4_grad(double /*xi*/, double /*eta*/, double /*zeta*/) {
    MatrixXd dN(4, 3);
    dN(0, 0) = -1.0;  dN(0, 1) = -1.0;  dN(0, 2) = -1.0;
    dN(1, 0) =  1.0;  dN(1, 1) =  0.0;  dN(1, 2) =  0.0;
    dN(2, 0) =  0.0;  dN(2, 1) =  1.0;  dN(2, 2) =  0.0;
    dN(3, 0) =  0.0;  dN(3, 1) =  0.0;  dN(3, 2) =  1.0;
    return dN;
}

void ElementLibrary3D::register_builtins_() {
    // -------------------------------------------------------------------------
    // Hex8: 2×2×2 primary integration, 3×3×3 higher-order quadrature
    // -------------------------------------------------------------------------
    {
        ElementType3D et;
        et.name = "Hex8";
        et.n_nodes = 8;
        et.n_integration_points = 8;
        et.n_quad_points = 27;

        et.shape_functions    = hex8_shape;
        et.shape_fn_gradients = hex8_grad;

        // 2×2×2 primary Gauss rule (±1/√3), all weights = 1.0
        const double gp2 = 1.0 / std::sqrt(3.0);
        const double xi2[2]   = {-gp2,  gp2};
        const double eta2[2]  = {-gp2,  gp2};
        const double zeta2[2] = {-gp2,  gp2};
        for (int k = 0; k < 2; ++k)
            for (int j = 0; j < 2; ++j)
                for (int i = 0; i < 2; ++i) {
                    et.gauss_pts_natural.push_back({xi2[i], eta2[j], zeta2[k]});
                    et.gauss_weights.push_back(1.0);
                }

        // 3×3×3 higher-order Gauss rule (±√(3/5), 0) for computing V and M
        const double gp3 = std::sqrt(0.6);
        const double xi3[3]  = {-gp3, 0.0, gp3};
        const double we3[3]  = {5.0/9.0, 8.0/9.0, 5.0/9.0};
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i) {
                    et.quad_pts_natural.push_back({xi3[i], xi3[j], xi3[k]});
                    et.quad_weights.push_back(we3[i] * we3[j] * we3[k]);
                }

        register_type(et);
    }

    // -------------------------------------------------------------------------
    // Tet4: 4-point primary rule, 4-point quadrature (linear — no higher order needed)
    // -------------------------------------------------------------------------
    {
        ElementType3D et;
        et.name = "Tet4";
        et.n_nodes = 4;
        et.n_integration_points = 4;
        et.n_quad_points = 4;

        et.shape_functions    = tet4_shape;
        et.shape_fn_gradients = tet4_grad;

        // Standard 4-point Gauss rule for tetrahedron, exact for degree 2.
        // Reference tet: vertices at (0,0,0),(1,0,0),(0,1,0),(0,0,1), volume=1/6.
        // a = (5-sqrt(5))/20 ≈ 0.1381966011250105
        // b = 1-3a           ≈ 0.5854101966249685
        // All weights = 1/24 (= volume/4).
        const double a = (5.0 - std::sqrt(5.0)) / 20.0;
        const double b = 1.0 - 3.0 * a;
        const double w = 1.0 / 24.0;

        et.gauss_pts_natural = {
            {a, a, a},
            {b, a, a},
            {a, b, a},
            {a, a, b},
        };
        et.gauss_weights = { w, w, w, w };

        // Same rule used for higher-order quadrature (linear element, so no gain
        // from a denser rule; upgrade to Keast 5-point if needed).
        et.quad_pts_natural = et.gauss_pts_natural;
        et.quad_weights     = et.gauss_weights;

        register_type(et);
    }
}

} // namespace l2map
