#pragma once
#include "types.hpp"
#include <string>
#include <functional>
#include <unordered_map>
#include <stdexcept>

namespace l2map {

enum class Dimension { D2, D3 };
enum class PointRole { INTEGRATION_POINT, NODE };

struct ElementType {
    std::string name;
    Dimension   dim;
    int         n_nodes;
    int         n_integration_points;
    int         poly_degree;
    int         n_monomials;

    // Natural coordinates of integration points in reference element [-1,1]^d
    std::vector<Point2D> gauss_pts_natural;
    std::vector<double>  gauss_weights;

    // Natural coordinates of nodes in reference element
    std::vector<Point2D> node_pts_natural;

    // Maps storage order -> counterclockwise polygon traversal order
    // e.g. Quad8: {0, 4, 1, 5, 2, 6, 3, 7}
    std::vector<int> polygon_vertex_order;

    // Shape function evaluation at natural coord (xi, eta).
    // Returns vector of length n_nodes.
    std::function<VectorXd(double xi, double eta)> shape_functions;
};

class ElementLibrary {
public:
    static ElementLibrary& instance();

    void register_type(const ElementType& et);
    const ElementType& get(const std::string& name) const;

    // Compute global coords of all Gauss points for an element.
    // nodes_global: rows = nodes, cols = [x, y]
    std::vector<Point2D> compute_gauss_points_global(
        const std::string& type_name,
        const MatrixXd& nodes_global) const;

    // Compute the polygon vertices for the element in CCW order.
    Polygon2D element_polygon(
        const std::string& type_name,
        const MatrixXd& nodes_global) const;

private:
    ElementLibrary();
    std::unordered_map<std::string, ElementType> types_;
    void register_builtins_();
};

} // namespace l2map
