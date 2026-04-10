#pragma once
#include "types.hpp"
#include <vector>
#include <tuple>
#include <stdexcept>

namespace l2map {

// ---------------------------------------------------------------------------
// 3D monomial basis descriptor.
// Ordering: tensor-product for k³ points: {x^i y^j z^k : 0≤i,j,k≤m-1}
// where m = cbrt(n_points). Sorted by total degree then lexicographically.
//
// For n=8  (k=2): {1, x, y, z, xy, xz, yz, xyz}          (max degree 3)
// For n=27 (k=3): {x^i y^j z^k : 0≤i,j,k≤2}               (max degree 6)
// ---------------------------------------------------------------------------

struct MonomialBasis3D {
    int n_monomials;
    // Each monomial is (power_x, power_y, power_z)
    std::vector<std::tuple<int, int, int>> monomials;

    // Evaluate all monomials at (x, y, z); returns vector of length n_monomials
    VectorXd evaluate(double x, double y, double z) const;

    // Degree of monomial at index i
    int degree(int i) const {
        auto [px, py, pz] = monomials[i];
        return px + py + pz;
    }
    int max_degree() const {
        if (monomials.empty()) return 0;
        auto [px, py, pz] = monomials.back();
        return px + py + pz;
    }
};

// Get tensor-product monomial basis for N = k³ points.
// For k=2 (N=8):  {1, x, y, z, xy, xz, yz, xyz}
// For k=3 (N=27): {x^i y^j z^k : 0≤i,j,k≤2}, ordered by total degree.
// Falls back to partial Pascal tetrahedron if N is not a perfect cube.
MonomialBasis3D get_tensor_basis_3d(int n_points);

// ---------------------------------------------------------------------------
// 3D basis builder: builds Lagrange polynomials from a set of 3D points.
// ---------------------------------------------------------------------------

class BasisBuilder3D {
public:
    // Build basis polynomials for a set of N 3D points.
    // Returns BasisMatrix of shape (N, N):
    //   row i = coefficient vector of φ_i s.t. φ_i(x_j) = δ_{ij}
    //
    // Points should be shifted (last point near origin) for conditioning.
    // Throws std::runtime_error if Vandermonde matrix is near-singular.
    BasisMatrix build(const std::vector<Point3D>& points) const;

    // Evaluate the i-th basis polynomial at point p given basis matrix and monomial descriptor
    double evaluate_basis(const BasisMatrix& basis,
                          int i,
                          const Point3D& p,
                          const MonomialBasis3D& mono) const;

private:
    MatrixXd build_vandermonde_(const std::vector<Point3D>& points,
                                const MonomialBasis3D& mono) const;
};

} // namespace l2map
