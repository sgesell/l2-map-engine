#pragma once
#include "types.hpp"
#include <vector>
#include <utility>
#include <stdexcept>

namespace l2map {

// Monomial basis descriptor for 2D.
// Ordering follows Pascal triangle: degree 0, then 1, then 2, ...
// Within each degree: decreasing x power, increasing y power.
// e.g. degree 2: (2,0), (1,1), (0,2)  →  x², xy, y²
struct MonomialBasis2D {
    int n_monomials;
    // Each monomial is (power_x, power_y)
    std::vector<std::pair<int, int>> monomials;

    // Evaluate all monomials at (x, y); returns vector of length n_monomials
    VectorXd evaluate(double x, double y) const;

    // Degree of monomial at index i
    int degree(int i) const { return monomials[i].first + monomials[i].second; }

    // Maximum degree present
    int max_degree() const {
        if (monomials.empty()) return 0;
        return monomials.back().first + monomials.back().second;
    }
};

// Get the canonical monomial basis for N points in 2D.
// Fills the Pascal triangle row by row until N monomials.
MonomialBasis2D get_monomial_basis_2d(int n_points);

// Get the tensor-product (serendipity) monomial basis for N = k² points.
// For N = k², returns {x^i * y^j : 0 ≤ i,j ≤ k-1}, ordered by total degree
// then by decreasing x-power within each degree.  For k=3 (N=9) this gives
// {1, x, y, x², xy, y², x²y, xy², x²y²} — the biquadratic basis that is
// unisolvent on any k×k grid of distinct points.
// Falls back to get_monomial_basis_2d for non-perfect-square N.
MonomialBasis2D get_serendipity_basis_2d(int n_points);

class BasisBuilder {
public:
    // Build basis polynomials for a set of N points.
    // Returns BasisMatrix of shape (N, N):
    //   row i = coefficient vector of the i-th Lagrange basis polynomial φ_i
    //   such that φ_i(x_j, y_j) = δ_{ij}
    //
    // IMPORTANT: points should be shifted (last point near origin) for conditioning.
    // This function does NOT shift — caller is responsible.
    //
    // Throws std::runtime_error if Vandermonde matrix is near-singular.
    BasisMatrix build(const std::vector<Point2D>& points) const;

    // Evaluate the i-th basis polynomial at point p.
    double evaluate_basis(const BasisMatrix& basis,
                          int i,
                          const Point2D& p,
                          const MonomialBasis2D& mono) const;

private:
    MatrixXd build_vandermonde_(const std::vector<Point2D>& points,
                                const MonomialBasis2D& mono) const;
};

} // namespace l2map
