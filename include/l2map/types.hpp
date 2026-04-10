#pragma once
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <unordered_map>
#include <cstdint>

namespace l2map {

// Spatial points
using Point2D = Eigen::Vector2d;
using Point3D = Eigen::Vector3d;

// A polygon is an ordered sequence of 2D vertices.
// First and last vertex are NOT repeated (closed implicitly).
using Polygon2D = std::vector<Point2D>;

// Dense matrix and vector aliases
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

// ID types (1-indexed in files, converted to 0-indexed internally)
using NodeID = int32_t;
using ElemID = int32_t;

// Polynomial coefficient vector.
// Ordering: Pascal triangle row-by-row:
//   [c0(1), c1(x), c2(y), c3(x²), c4(xy), c5(y²), c6(x²y), c7(xy²), ...]
using PolyCoeffs = VectorXd;

// Basis polynomial matrix: rows = one basis polynomial per integration point,
// cols = monomial coefficients. Shape: (N_pts × N_monomials).
using BasisMatrix = MatrixXd;

// Field data for one element: shape (N_pts_per_element × N_components)
using FieldSlice = MatrixXd;

// Cache types (2D)
using IntPointCache  = std::unordered_map<ElemID, std::vector<Point2D>>;
using BasisCache     = std::unordered_map<ElemID, BasisMatrix>;
using FieldDataCache = std::unordered_map<ElemID, FieldSlice>;

// Cache types (3D)
using IntPointCache3D    = std::unordered_map<ElemID, std::vector<Point3D>>;
using BasisCache3D       = std::unordered_map<ElemID, BasisMatrix>;
// Per-element quadrature weights in PHYSICAL space: w_q * |J(xi_q)|
// Shape: vector of length n_quad_pts
using QuadWeightCache3D  = std::unordered_map<ElemID, std::vector<double>>;
// Quadrature points in physical space (higher-order rule for computing V and M)
using QuadPointCache3D   = std::unordered_map<ElemID, std::vector<Point3D>>;

} // namespace l2map
