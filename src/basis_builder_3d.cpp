#include "l2map/basis_builder_3d.hpp"
#include <cmath>
#include <sstream>

namespace l2map {

// ---------------------------------------------------------------------------
// MonomialBasis3D
// ---------------------------------------------------------------------------

VectorXd MonomialBasis3D::evaluate(double x, double y, double z) const {
    VectorXd v(n_monomials);
    for (int i = 0; i < n_monomials; ++i) {
        auto [px, py, pz] = monomials[i];
        double val = 1.0;
        for (int k = 0; k < px; ++k) val *= x;
        for (int k = 0; k < py; ++k) val *= y;
        for (int k = 0; k < pz; ++k) val *= z;
        v[i] = val;
    }
    return v;
}

MonomialBasis3D get_tensor_basis_3d(int n_points) {
    // Determine k = round(cbrt(n_points))
    int k = static_cast<int>(std::round(std::cbrt(static_cast<double>(n_points))));

    MonomialBasis3D basis;
    basis.n_monomials = n_points;
    basis.monomials.reserve(n_points);

    if (k * k * k == n_points) {
        // Perfect cube: tensor product basis {x^i y^j z^l : 0≤i,j,l≤k-1}
        // Order by total degree, then lexicographic within each degree
        for (int deg = 0; static_cast<int>(basis.monomials.size()) < n_points; ++deg) {
            // Enumerate all (px,py,pz) with px+py+pz == deg and 0<=px,py,pz<=k-1
            for (int px = std::min(deg, k-1); px >= 0; --px) {
                for (int py = std::min(deg - px, k-1); py >= 0; --py) {
                    int pz = deg - px - py;
                    if (pz < 0 || pz > k - 1) continue;
                    basis.monomials.push_back({px, py, pz});
                    if (static_cast<int>(basis.monomials.size()) == n_points) goto done;
                }
            }
        }
        done:;
    } else {
        // Non-perfect-cube: Pascal tetrahedron order
        for (int deg = 0; static_cast<int>(basis.monomials.size()) < n_points; ++deg) {
            for (int px = deg; px >= 0; --px) {
                for (int py = deg - px; py >= 0; --py) {
                    int pz = deg - px - py;
                    basis.monomials.push_back({px, py, pz});
                    if (static_cast<int>(basis.monomials.size()) == n_points) goto done2;
                }
            }
        }
        done2:;
    }
    return basis;
}

// ---------------------------------------------------------------------------
// BasisBuilder3D
// ---------------------------------------------------------------------------

MatrixXd BasisBuilder3D::build_vandermonde_(const std::vector<Point3D>& points,
                                            const MonomialBasis3D& mono) const
{
    int N = static_cast<int>(points.size());
    MatrixXd A(N, N);
    for (int i = 0; i < N; ++i) {
        A.row(i) = mono.evaluate(points[i][0], points[i][1], points[i][2]);
    }
    return A;
}

BasisMatrix BasisBuilder3D::build(const std::vector<Point3D>& points) const {
    int N = static_cast<int>(points.size());
    if (N == 0) return BasisMatrix(0, 0);

    // Shift so last point becomes origin (numerical conditioning)
    Point3D origin = points.back();
    std::vector<Point3D> shifted(N);
    for (int i = 0; i < N; ++i)
        shifted[i] = points[i] - origin;

    MonomialBasis3D mono = get_tensor_basis_3d(N);
    MatrixXd A = build_vandermonde_(shifted, mono);

    // ColPivHouseholderQR provides rank detection and is substantially faster
    // than FullPivLU for the well-conditioned Vandermonde matrices we build from
    // shifted Gauss points.
    Eigen::ColPivHouseholderQR<MatrixXd> qr(A);
    if (qr.rank() < N) {
        std::ostringstream ss;
        ss << "BasisBuilder3D: Vandermonde matrix rank-deficient (rank=" << qr.rank()
           << " < " << N << ") for " << N << " 3D points. "
           << "First point: (" << points[0][0] << ", " << points[0][1]
           << ", " << points[0][2] << "). "
           << "This typically indicates a degenerate element geometry.";
        throw std::runtime_error(ss.str());
    }

    // Column j of A^{-1} = coefficients of φ_j.
    // Return A^{-T} so row j = coefficients of φ_j.
    MatrixXd I = MatrixXd::Identity(N, N);
    return qr.solve(I).transpose();  // shape (N, N)
}

double BasisBuilder3D::evaluate_basis(const BasisMatrix& basis,
                                      int i,
                                      const Point3D& p,
                                      const MonomialBasis3D& mono) const
{
    VectorXd m = mono.evaluate(p[0], p[1], p[2]);
    return basis.row(i).dot(m);
}

} // namespace l2map
