#ifndef STRATEGIES_H
#define STRATEGIES_H

#include <complex>

namespace strategies {
void
compute_beamsplitter(int M, int N, int P, int Q, double theta, double phi, std::complex<double> *G);

void
compute_beamsplitter_stable(
    int M, int N, int P, int Q, double theta, double phi, std::complex<double> *G
);

void
compute_displacement(
    int N, int M, std::complex<double> alpha, bool flipped, std::complex<double> *D
);

void
compute_homodyne_projector(
    int M,
    int N,
    int P,
    const std::complex<double> *A,
    const std::complex<double> *b,
    const std::complex<double> c,
    std::complex<double> *G
);

void
compute_jacobian_displacement(
    int N,
    int M,
    std::complex<double> alpha,
    const std::complex<double> *D,
    std::complex<double> *jac_alpha,
    std::complex<double> *jac_alphac
);

void
compute_squeezed(int N, double r, double theta, std::complex<double> *S);

void
compute_squeezer(int M, int N, double r, double theta, std::complex<double> *S);
}  // namespace strategies

#endif