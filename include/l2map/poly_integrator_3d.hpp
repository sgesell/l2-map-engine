#pragma once
#include "types.hpp"
#include "basis_builder_3d.hpp"
#include "polyhedron_clipper.hpp"
#include "poly_integrator.hpp"   // reused for 2D face integrals
#include "basis_builder.hpp"     // MonomialBasis2D

namespace l2map {

// ---------------------------------------------------------------------------
// PolyIntegrator3D — integrate polynomials EXACTLY over convex polyhedra.
//
// Uses the divergence theorem reduction:
//   ∫_P g dV  =  Σ_d  (1/(d+3))  Σ_{faces k}  (n_k · v_0k)  *  ∫_{face k} g_d dA
//
// where:
//   g_d     = degree-d homogeneous part of g
//   n_k     = outward unit normal of face k
//   v_0k    = any vertex on face k  (n_k · v_0k is constant on the face plane)
//   ∫_{face} g_d dA  = computed in local face (u,v) coordinates using the
//                       existing 2D PolyIntegrator (Stokes theorem on edges)
//
// Accuracy: exact for any polynomial integrand (up to floating-point rounding).
// ---------------------------------------------------------------------------

class PolyIntegrator3D {
public:
    // n_gauss_pts: points per edge in the underlying 1D Gauss rule (passed to 2D integrator)
    explicit PolyIntegrator3D(int n_gauss_pts = 5);

    // Integrate a 3D polynomial over a convex polyhedron.
    // coeffs: coefficients in MonomialBasis3D ordering
    // mono:   the 3D monomial descriptor matching coeffs
    double integrate(const Polyhedron& poly,
                     const VectorXd& coeffs,
                     const MonomialBasis3D& mono) const;

    // Multiply two 3D polynomials.
    // mono_product_out is filled with the product basis.
    VectorXd multiply_polynomials(const VectorXd& ca, const MonomialBasis3D& ma,
                                  const VectorXd& cb, const MonomialBasis3D& mb,
                                  MonomialBasis3D& mono_product_out) const;

private:
    PolyIntegrator integrator_2d_;  // Gauss-Legendre for 1D edge integrals

    // Decompose a 3D polynomial into homogeneous parts.
    // Returns vector indexed by degree d; each entry is a VectorXd of length N_d
    // where N_d = number of degree-d monomials present in mono.
    // Format of each entry: dense coefficient for ALL degree-d monomials in Pascal order.
    struct HomoPart {
        int degree;
        // coefficients indexed by position WITHIN that degree:
        // For degree d: monomials (px,py,pz) with px+py+pz=d.
        // Stored as flat array of length matching get_tensor_basis_3d monomials at that degree.
        std::vector<std::pair<std::tuple<int,int,int>, double>> terms;
    };
    std::vector<HomoPart> decompose_homogeneous_(const VectorXd& coeffs,
                                                 const MonomialBasis3D& mono) const;

    // Integrate ONE homogeneous 3D polynomial over ONE planar face (polygon in 3D).
    // Reduces to 2D Stokes on the face plane.
    double integrate_homo_on_face_(const HomoPart& hp, const Face3D& face) const;

    // Expand (c0 + cu*u + cv*v)^n into Pascal-ordered 2D polynomial coefficients.
    // Returns VectorXd of size (n+1)*(n+2)/2.
    static VectorXd expand_linear_power_(double c0, double cu, double cv, int n);

    // Multiply two Pascal-ordered 2D polynomials of given degrees.
    // Returns product coefficients in Pascal order.
    static VectorXd multiply_poly2d_(const VectorXd& a, int deg_a,
                                     const VectorXd& b, int deg_b);
};

} // namespace l2map
