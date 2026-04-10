#pragma once
#include "types.hpp"
#include <string>
#include <functional>
#include <unordered_map>
#include <array>

namespace l2map {

// -----------------------------------------------------------------------------
// 3D element type descriptor
// -----------------------------------------------------------------------------
struct ElementType3D {
    std::string name;
    int n_nodes;
    int n_integration_points;  // primary (element) Gauss rule
    int n_quad_points;         // higher-order rule for computing V and M

    // Primary integration points in reference space [-1,1]³: (xi,eta,zeta)
    std::vector<std::array<double, 3>> gauss_pts_natural;
    std::vector<double>                gauss_weights;

    // Higher-order quadrature in reference space (n_quad_points entries)
    std::vector<std::array<double, 3>> quad_pts_natural;
    std::vector<double>                quad_weights;

    // Shape functions: N(xi,eta,zeta) -> vector of length n_nodes
    std::function<VectorXd(double xi, double eta, double zeta)> shape_functions;

    // Shape function gradients: dN/d(xi,eta,zeta) -> matrix (n_nodes x 3)
    // Column 0: dN/dxi, Column 1: dN/deta, Column 2: dN/dzeta
    std::function<MatrixXd(double xi, double eta, double zeta)> shape_fn_gradients;
};

// -----------------------------------------------------------------------------
// 3D element library (singleton)
// -----------------------------------------------------------------------------
class ElementLibrary3D {
public:
    static ElementLibrary3D& instance();

    void register_type(const ElementType3D& et);
    const ElementType3D& get(const std::string& name) const;

    // Compute physical coords of primary Gauss points for an element.
    // node_coords: (n_nodes x 3) matrix [x,y,z]
    std::vector<Point3D> compute_gauss_points_global(
        const std::string& type_name,
        const MatrixXd& node_coords) const;

    // Compute physical coords of higher-order quadrature points + physical weights.
    // physical_weights[q] = quad_weights[q] * |J(xi_q)|
    void compute_quad_points_global(
        const std::string& type_name,
        const MatrixXd& node_coords,
        std::vector<Point3D>& phys_pts_out,
        std::vector<double>&  phys_weights_out) const;

    // Compute Jacobian determinant at a reference point
    double jacobian_det(const std::string& type_name,
                        const MatrixXd& node_coords,
                        double xi, double eta, double zeta) const;

private:
    ElementLibrary3D();
    std::unordered_map<std::string, ElementType3D> types_;
    void register_builtins_();
};

} // namespace l2map
