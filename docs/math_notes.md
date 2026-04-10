# L2MapEngine — Math Notes

## Core equation

The L²-projection satisfies:

$$\int_\Omega u_h \, \chi \, dV = \int_\Omega u_n \, \chi \, dV \qquad \forall \, \chi \in V_n$$

where $u_h$ is the source field and $u_n$ is the projected field on the new mesh.

## Element-local system

For each new element $e_n$, the local system is:

$$\mathbf{V} \boldsymbol{\alpha} = \mathbf{M}$$

$$V_{ij} = \int_{e_n} \varphi_i \, \varphi_j \, dA$$

$$M_{jl} = \sum_{\substack{e_o \,:\, e_o \cap e_n \neq \emptyset}}
           \int_{e_o \cap e_n} \varphi_j \left( \sum_k \beta_{kl} \, \psi_k \right) dA$$

where:
- $\varphi_i$ are basis polynomials for the new element
- $\psi_k$ are basis polynomials for the old element
- $\beta_{kl}$ are field values at old element integration points (component $l$)
- $\alpha_{il}$ are the solved field values at new element integration points

## Stokes theorem for polygon integration

$$\int_\Omega g(x,y) \, dA
  = \sum_d \frac{1}{d+2}
    \sum_{\text{edges}} \int_0^1 f_d\!\left(F(t)\right) dt$$

where $f_d$ is the homogeneous degree-$d$ component of $g$, and
$F(t) = v_k + t(v_{k+1} - v_k)$ is the linear parametrisation of each edge.

## Coordinate shifting

All computations shift coordinates so that the last Gauss point of the **new** element is at the origin:

$$x_{\text{shifted}} = x_{\text{global}} - x_{\text{origin}}$$

This is critical for numerical stability when absolute coordinates are large.

## Monomial basis (Pascal triangle order)

| Degree | Monomials | Terms | Running total |
|--------|-----------|-------|---------------|
| 0 | $1$ | 1 | 1 |
| 1 | $x,\ y$ | 2 | 3 |
| 2 | $x^2,\ xy,\ y^2$ | 3 | 6 |
| 3 | $x^3,\ x^2y,\ xy^2,\ y^3$ | 4 | 10 |
| 4 | $x^4,\ x^3y,\ x^2y^2,\ xy^3,\ y^4$ | 5 | 15 |

For a Quad8 element with 9 integration points, the library uses the first 9 monomials in Pascal order:
$\{1,\, x,\, y,\, x^2,\, xy,\, y^2,\, x^3,\, x^2y,\, xy^2\}$.

> **Note:** this is *not* the serendipity 8-monomial set. The Python reference implementation also uses 9 monomials.

## Vandermonde inversion

The Vandermonde matrix is evaluated at the shifted Gauss points $(\hat{x}_i, \hat{y}_i)$:

$$[A_P]_{ij} = m_j(\hat{x}_i,\, \hat{y}_i), \qquad U_P = A_P^{-1}$$

computed via `Eigen::FullPivLU`. Row $k$ of $U_P$ gives the coefficient vector of basis polynomial $\varphi_k$.

## References

- Gesell, S. (2024). "Anwendung des CTOD-Konzepts auf Rissfortschritt unter thermomechanischer Beanspruchung mithilfe von Experimenten und numerischer Simulation." Doctoral thesis, Technische Universität Bergakademie Freiberg. URN: [urn:nbn:de:bsz:105-qucosa2-928920](https://nbn-resolving.org/urn:nbn:de:bsz:105-qucosa2-928920)
- Chin, E.B., Lasserre, J.B., Sukumar, N. (2015). "Numerical integration of homogeneous functions on convex and nonconvex polygons and polyhedra." *Computational Mechanics*.
- ABAQUS Theory Manual, Section on element formulations.
