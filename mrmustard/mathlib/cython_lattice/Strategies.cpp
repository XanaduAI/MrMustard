#include "Strategies.hpp"

#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

namespace utils {
constexpr int SQRT_SIZE = 100000;
static double SQRT[SQRT_SIZE];
static double INV_SQRT[SQRT_SIZE];

static bool sqrt_arrays_initialized = []() {
    for (int i = 0; i < SQRT_SIZE; i++) {
        SQRT[i] = std::sqrt(static_cast<double>(i));
        INV_SQRT[i] = (i == 0) ? 0.0 : 1.0 / SQRT[i];
    }
    return true;
}();

static void
laguerre(double x, int N, double alpha, std::complex<double> *ret)
{
    ret[0] = std::complex<double>(1, 0);
    if (N > 1) {
        ret[1] = (1 + alpha - x) * ret[0];
        for (int m = 1; m < (N - 1); m++) {
            ret[m + 1] = ((2 * m + 1 + alpha - x) * ret[m] - (m + alpha) * ret[m - 1]) /
                         std::complex<double>(m + 1, 0);
        }
    }
}
}  // namespace utils

namespace strategies {
void
compute_beamsplitter(int M, int N, int P, int Q, double theta, double phi, std::complex<double> *G)
{
    double ct = std::cos(theta);
    std::complex<double> st = std::exp(std::complex<double>(0, phi)) * std::sin(theta);
    std::complex<double> stc = std::conj(st);

    std::memset(G, 0, M * N * P * Q * sizeof(std::complex<double>));

    // Helper lambda for 4D array indexing: G[m][n][p][q]
    auto idx = [N, P, Q](int m, int n, int p, int q) -> int {
        return m * (N * P * Q) + n * (P * Q) + p * Q + q;
    };

    G[idx(0, 0, 0, 0)] = std::complex<double>(1.0, 0.0);

    // rank 3
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N - m; n++) {
            int p = m + n;
            if (0 < p && p < P) {
                std::complex<double> term1 =
                    (m > 0) ? ct * utils::SQRT[m] * utils::INV_SQRT[p] * G[idx(m - 1, n, p - 1, 0)]
                            : 0.0;
                std::complex<double> term2 =
                    (n > 0) ? st * utils::SQRT[n] * utils::INV_SQRT[p] * G[idx(m, n - 1, p - 1, 0)]
                            : 0.0;
                G[idx(m, n, p, 0)] = term1 + term2;
            }
        }
    }

    // rank 4
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            for (int p = 0; p < P; p++) {
                int q = m + n - p;
                if (0 < q && q < Q) {
                    std::complex<double> term1 = (m > 0)
                                                     ? -stc * utils::SQRT[m] * utils::INV_SQRT[q] *
                                                           G[idx(m - 1, n, p, q - 1)]
                                                     : 0.0;
                    std::complex<double> term2 =
                        (n > 0)
                            ? ct * utils::SQRT[n] * utils::INV_SQRT[q] * G[idx(m, n - 1, p, q - 1)]
                            : 0.0;
                    G[idx(m, n, p, q)] = term1 + term2;
                }
            }
        }
    }
}

void
compute_beamsplitter_stable(
    int M, int N, int P, int Q, double theta, double phi, std::complex<double> *G
)
{
    double ct = std::cos(theta);
    double st_mag = std::sin(theta);
    std::complex<double> st = st_mag * (std::cos(phi) + std::complex<double>(0, 1) * std::sin(phi));
    std::complex<double> stc =
        st_mag * (std::cos(phi) - std::complex<double>(0, 1) * std::sin(phi));

    // Zero out the entire array using memset (much faster than nested loops)
    std::memset(G, 0, M * N * P * Q * sizeof(std::complex<double>));

    // Helper lambda for 4D array indexing: G[m][n][p][q]
    auto idx = [N, P, Q](int m, int n, int p, int q) -> int {
        return m * (N * P * Q) + n * (P * Q) + p * Q + q;
    };

    G[idx(0, 0, 0, 0)] = std::complex<double>(1.0, 0.0);

    std::complex<double> val;
    int pivots;
    double inv_num_pivots;

    // rank 3
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < std::min(N, P - m); n++) {
            int p = m + n;
            val = std::complex<double>(0, 0);
            pivots = 0;
            if (m > 0 && p > 0) {  // pivot at (m-1, n, p, 0)
                val += ct * utils::SQRT[p] * utils::INV_SQRT[m] * G[idx(m - 1, n, p - 1, 0)];
                pivots += 1;
            }
            if (n > 0 && p > 0) {  // pivot at (m, n-1, p, 0)
                val += st * utils::SQRT[p] * utils::INV_SQRT[n] * G[idx(m, n - 1, p - 1, 0)];
                pivots += 1;
            }
            if (p > 0) {  // pivot at (m, n, p-1, 0)
                std::complex<double> pivot_val(0, 0);
                if (m > 0) {
                    pivot_val +=
                        ct * utils::SQRT[m] * utils::INV_SQRT[p] * G[idx(m - 1, n, p - 1, 0)];
                }
                if (n > 0) {
                    pivot_val +=
                        st * utils::SQRT[n] * utils::INV_SQRT[p] * G[idx(m, n - 1, p - 1, 0)];
                }
                if (m > 0 || n > 0) {
                    val += pivot_val;
                    pivots += 1;
                }
            }
            if (m > 0 || n > 0 || p > 0) {
                inv_num_pivots = 1.0 / pivots;
                G[idx(m, n, p, 0)] = val * inv_num_pivots;
            }
        }
    }

    // rank 4
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            for (int p = std::max(0, m + n - Q); p < std::min(P, m + n); p++) {
                int q = m + n - p;
                if (0 < q && q < Q) {
                    val = std::complex<double>(0, 0);
                    pivots = 0;
                    if (m > 0) {
                        std::complex<double> pivot_val(0, 0);
                        if (p > 0) {
                            pivot_val += ct * utils::SQRT[p] * utils::INV_SQRT[m] *
                                         G[idx(m - 1, n, p - 1, q)];
                        }
                        if (q > 0) {
                            pivot_val += -stc * utils::SQRT[q] * utils::INV_SQRT[m] *
                                         G[idx(m - 1, n, p, q - 1)];
                        }
                        if (p > 0 || q > 0) {
                            val += pivot_val;
                            pivots += 1;
                        }
                    }
                    if (n > 0) {
                        std::complex<double> pivot_val(0, 0);
                        if (p > 0) {
                            pivot_val += st * utils::SQRT[p] * utils::INV_SQRT[n] *
                                         G[idx(m, n - 1, p - 1, q)];
                        }
                        if (q > 0) {
                            pivot_val += ct * utils::SQRT[q] * utils::INV_SQRT[n] *
                                         G[idx(m, n - 1, p, q - 1)];
                        }
                        if (p > 0 || q > 0) {
                            val += pivot_val;
                            pivots += 1;
                        }
                    }
                    if (p > 0) {
                        std::complex<double> pivot_val(0, 0);
                        if (m > 0) {
                            pivot_val += ct * utils::SQRT[m] * utils::INV_SQRT[p] *
                                         G[idx(m - 1, n, p - 1, q)];
                        }
                        if (n > 0) {
                            pivot_val += st * utils::SQRT[n] * utils::INV_SQRT[p] *
                                         G[idx(m, n - 1, p - 1, q)];
                        }
                        if (m > 0 || n > 0) {
                            val += pivot_val;
                            pivots += 1;
                        }
                    }
                    if (q > 0) {
                        std::complex<double> pivot_val(0, 0);
                        if (m > 0) {
                            pivot_val += -stc * utils::SQRT[m] * utils::INV_SQRT[q] *
                                         G[idx(m - 1, n, p, q - 1)];
                        }
                        if (n > 0) {
                            pivot_val += ct * utils::SQRT[n] * utils::INV_SQRT[q] *
                                         G[idx(m, n - 1, p, q - 1)];
                        }
                        if (m > 0 || n > 0) {
                            val += pivot_val;
                            pivots += 1;
                        }
                    }
                    if (m > 0 || n > 0 || p > 0 || q > 0) {
                        inv_num_pivots = 1.0 / pivots;
                        G[idx(m, n, p, q)] = val * inv_num_pivots;
                    }
                }
            }
        }
    }
}

void
compute_displacement(
    int N, int M, std::complex<double> alpha, bool flipped, std::complex<double> *D
)
{
    double r = std::abs(alpha);

    // Handle alpha=0 case: D(0) = Identity matrix
    if (r == 0.0) {
        // Initialize all elements to zero
        for (int i = 0; i < N * M; i++) {
            D[i] = std::complex<double>(0, 0);
        }
        // Set diagonal elements to 1
        int diag_size = std::min(N, M);
        for (int i = 0; i < diag_size; i++) {
            D[i * M + i] = std::complex<double>(1, 0);
        }
        return;
    }

    double r_squared = r * r;
    double phi = std::arg(alpha);

    std::vector<double> log_k_fac(N);
    log_k_fac[0] = 0.0;

    for (int i = 1; i < N; i++) {
        log_k_fac[i] = log_k_fac[i - 1] + std::log(double(i));
    }

    std::vector<std::complex<double>> L(M);
    std::vector<std::complex<double>> logL(M);

    for (int n_minus_m = 0; n_minus_m < N; n_minus_m++) {
        int m_max = std::min(M, N - n_minus_m);

        utils::laguerre(r_squared, m_max, n_minus_m, &L[0]);
        for (int i = 0; i < m_max; i++) {
            logL[i] = std::log(L[i]);
        }

        std::complex<double> alternating_sign = std::complex<double>((n_minus_m & 1) ? -1 : 1, 0);

        for (int m = 0; m < m_max; m++) {
            int n = n_minus_m + m;
            std::complex<double> sign =
                std::complex<double>(2 * (!(flipped && n > m && n_minus_m % 2)) - 1, 0);
            int conjugate = 2 * (!(flipped && n > m)) - 1;
            int idx_nm = n * M + m;
            D[idx_nm] = sign * std::exp(
                                   0.5 * (log_k_fac[m] - log_k_fac[n]) + n_minus_m * std::log(r) -
                                   r_squared / 2 +
                                   std::complex<double>(0, conjugate * phi * n_minus_m) + logL[m]
                               );
            if (n < M) {
                int idx_mn = m * M + n;
                D[idx_mn] = alternating_sign * std::conj(D[idx_nm]);
            }
        }
    }
}

void
compute_homodyne_projector(
    int M,
    int N,
    int P,
    const std::complex<double> *A,
    const std::complex<double> *b,
    const std::complex<double> c,
    std::complex<double> *G
)
{
    const int stride_0 = N * P;
    const int stride_1 = P;
    const int stride_2 = 1;
    const int size = M * N * P;

    // Precomputed reciprocals to avoid integer division (num_pivots is always 1,
    // 2, or 3)
    static constexpr double INV_PIVOTS[4] = {0.0, 1.0, 0.5, 1.0 / 3.0};

    // Initialize all elements to zero using memset (faster than loop)
    std::memset(G, 0, size * sizeof(std::complex<double>));
    G[0] = c;

    if (size == 1) {
        return;
    }

    // Pre-extract coefficients (marked const for compiler optimization)
    const std::complex<double> b1 = b[1], b2 = b[2];  // b[0] = 0
    const std::complex<double> A01 = A[0 * 3 + 1], A02 = A[0 * 3 + 2];
    const std::complex<double> A11 = A[1 * 3 + 1], A12 = A[1 * 3 + 2];
    const std::complex<double> A22 = A[2 * 3 + 2];  // A[0,0] = 0

    // Handle first element separately (flat_index = 1)
    int i = 0, j = 0, k = 1;
    if (size > 1) {
        G[1] = b2 * c;  // Only one valid pivot for (0,0,1)
    }

    int flat_index = 2;
    int pivot;
    std::complex<double> val;
    std::complex<double> total_value;
    int num_pivots;
    int current_idx;

    // Helper lambda for safe array access (returns 0 for negative indices)
    auto safe_get = [&G](int idx) -> std::complex<double> {
        return (idx >= 0) ? G[idx] : std::complex<double>(0.0, 0.0);
    };

    while (flat_index + 4 <= size) {
        // ===== ELEMENT 1 (flat_index + 0) =====

        k++;
        if (k >= P) {
            k = 0;
            j++;
            if (j >= N) {
                j = 0;
                i++;
            }
        }

        current_idx = flat_index;
        num_pivots = (i > 0) + (j > 0) + (k > 0);  // Branchless pivot count
        total_value = std::complex<double>(0.0, 0.0);

        if (i > 0) {
            pivot = current_idx - stride_0;
            val = A01 * utils::SQRT[j] * safe_get(pivot - stride_1) +
                  A02 * utils::SQRT[k] * safe_get(pivot - stride_2);
            total_value += val * utils::INV_SQRT[i];
        }

        if (j > 0) {
            pivot = current_idx - stride_1;
            val = b1 * G[pivot] + A01 * utils::SQRT[i] * safe_get(pivot - stride_0) +
                  A11 * utils::SQRT[j - 1] * safe_get(pivot - stride_1) +
                  A12 * utils::SQRT[k] * safe_get(pivot - stride_2);
            total_value += val * utils::INV_SQRT[j];
        }

        if (k > 0) {
            pivot = current_idx - stride_2;
            val = b2 * G[pivot] + A02 * utils::SQRT[i] * safe_get(pivot - stride_0) +
                  A12 * utils::SQRT[j] * safe_get(pivot - stride_1) +
                  A22 * utils::SQRT[k - 1] * safe_get(pivot - stride_2);
            total_value += val * utils::INV_SQRT[k];
        }
        G[current_idx] =
            total_value * INV_PIVOTS[num_pivots];  // Single write with precomputed reciprocal

        // ===== ELEMENT 2 (flat_index + 1) =====

        k++;
        if (k >= P) {
            k = 0;
            j++;
            if (j >= N) {
                j = 0;
                i++;
            }
        }

        current_idx = flat_index + 1;
        num_pivots = (i > 0) + (j > 0) + (k > 0);
        total_value = std::complex<double>(0.0, 0.0);

        if (i > 0) {
            pivot = current_idx - stride_0;
            val = A01 * utils::SQRT[j] * safe_get(pivot - stride_1) +
                  A02 * utils::SQRT[k] * safe_get(pivot - stride_2);
            total_value += val * utils::INV_SQRT[i];
        }

        if (j > 0) {
            pivot = current_idx - stride_1;
            val = b1 * G[pivot] + A01 * utils::SQRT[i] * safe_get(pivot - stride_0) +
                  A11 * utils::SQRT[j - 1] * safe_get(pivot - stride_1) +
                  A12 * utils::SQRT[k] * safe_get(pivot - stride_2);
            total_value += val * utils::INV_SQRT[j];
        }

        if (k > 0) {
            pivot = current_idx - stride_2;
            val = b2 * G[pivot] + A02 * utils::SQRT[i] * safe_get(pivot - stride_0) +
                  A12 * utils::SQRT[j] * safe_get(pivot - stride_1) +
                  A22 * utils::SQRT[k - 1] * safe_get(pivot - stride_2);
            total_value += val * utils::INV_SQRT[k];
        }
        G[current_idx] = total_value * INV_PIVOTS[num_pivots];

        // ===== ELEMENT 3 (flat_index + 2) =====

        k++;
        if (k >= P) {
            k = 0;
            j++;
            if (j >= N) {
                j = 0;
                i++;
            }
        }

        current_idx = flat_index + 2;
        num_pivots = (i > 0) + (j > 0) + (k > 0);
        total_value = std::complex<double>(0.0, 0.0);

        if (i > 0) {
            pivot = current_idx - stride_0;
            val = A01 * utils::SQRT[j] * safe_get(pivot - stride_1) +
                  A02 * utils::SQRT[k] * safe_get(pivot - stride_2);
            total_value += val * utils::INV_SQRT[i];
        }

        if (j > 0) {
            pivot = current_idx - stride_1;
            val = b1 * G[pivot] + A01 * utils::SQRT[i] * safe_get(pivot - stride_0) +
                  A11 * utils::SQRT[j - 1] * safe_get(pivot - stride_1) +
                  A12 * utils::SQRT[k] * safe_get(pivot - stride_2);
            total_value += val * utils::INV_SQRT[j];
        }

        if (k > 0) {
            pivot = current_idx - stride_2;
            val = b2 * G[pivot] + A02 * utils::SQRT[i] * safe_get(pivot - stride_0) +
                  A12 * utils::SQRT[j] * safe_get(pivot - stride_1) +
                  A22 * utils::SQRT[k - 1] * safe_get(pivot - stride_2);
            total_value += val * utils::INV_SQRT[k];
        }
        G[current_idx] = total_value * INV_PIVOTS[num_pivots];

        // ===== ELEMENT 4 (flat_index + 3) =====
        k++;
        if (k >= P) {
            k = 0;
            j++;
            if (j >= N) {
                j = 0;
                i++;
            }
        }

        current_idx = flat_index + 3;
        num_pivots = (i > 0) + (j > 0) + (k > 0);
        total_value = std::complex<double>(0.0, 0.0);

        if (i > 0) {
            pivot = current_idx - stride_0;
            val = A01 * utils::SQRT[j] * safe_get(pivot - stride_1) +
                  A02 * utils::SQRT[k] * safe_get(pivot - stride_2);
            total_value += val * utils::INV_SQRT[i];
        }

        if (j > 0) {
            pivot = current_idx - stride_1;
            val = b1 * G[pivot] + A01 * utils::SQRT[i] * safe_get(pivot - stride_0) +
                  A11 * utils::SQRT[j - 1] * safe_get(pivot - stride_1) +
                  A12 * utils::SQRT[k] * safe_get(pivot - stride_2);
            total_value += val * utils::INV_SQRT[j];
        }

        if (k > 0) {
            pivot = current_idx - stride_2;
            val = b2 * G[pivot] + A02 * utils::SQRT[i] * safe_get(pivot - stride_0) +
                  A12 * utils::SQRT[j] * safe_get(pivot - stride_1) +
                  A22 * utils::SQRT[k - 1] * safe_get(pivot - stride_2);
            total_value += val * utils::INV_SQRT[k];
        }
        G[current_idx] = total_value * INV_PIVOTS[num_pivots];

        // Move to next group of 4
        flat_index += 4;
    }

    // Handle remaining elements (if any) with regular loop
    while (flat_index < size) {
        k++;
        if (k >= P) {
            k = 0;
            j++;
            if (j >= N) {
                j = 0;
                i++;
            }
        }

        num_pivots = (i > 0) + (j > 0) + (k > 0);
        total_value = std::complex<double>(0.0, 0.0);

        if (i > 0) {
            pivot = flat_index - stride_0;
            val = A01 * utils::SQRT[j] * safe_get(pivot - stride_1) +
                  A02 * utils::SQRT[k] * safe_get(pivot - stride_2);
            total_value += val * utils::INV_SQRT[i];
        }

        if (j > 0) {
            pivot = flat_index - stride_1;
            val = b1 * G[pivot] + A01 * utils::SQRT[i] * safe_get(pivot - stride_0) +
                  A11 * utils::SQRT[j - 1] * safe_get(pivot - stride_1) +
                  A12 * utils::SQRT[k] * safe_get(pivot - stride_2);
            total_value += val * utils::INV_SQRT[j];
        }

        if (k > 0) {
            pivot = flat_index - stride_2;
            val = b2 * G[pivot] + A02 * utils::SQRT[i] * safe_get(pivot - stride_0) +
                  A12 * utils::SQRT[j] * safe_get(pivot - stride_1) +
                  A22 * utils::SQRT[k - 1] * safe_get(pivot - stride_2);
            total_value += val * utils::INV_SQRT[k];
        }
        G[flat_index] = total_value * INV_PIVOTS[num_pivots];
        flat_index++;
    }
}

void
compute_jacobian_displacement(
    int N,
    int M,
    std::complex<double> alpha,
    const std::complex<double> *D,
    std::complex<double> *jac_alpha,
    std::complex<double> *jac_alphac
)
{
    std::complex<double> alphac = std::conj(alpha);
    for (int m = 0; m < N; m++) {
        for (int n = 0; n < M; n++) {
            jac_alpha[m * M + n] =
                -0.5 * alphac * D[m * M + n] + (m > 0 ? std::sqrt(m) * D[(m - 1) * M + n] : 0.0);
            jac_alphac[m * M + n] =
                -0.5 * alpha * D[m * M + n] - (n > 0 ? std::sqrt(n) * D[m * M + (n - 1)] : 0.0);
        }
    }
}

void
compute_squeezed(int N, double r, double theta, std::complex<double> *S)
{
    std::complex<double> eitheta_tanhr = std::exp(std::complex<double>(0, theta)) * -std::tanh(r);
    std::complex<double> sechr = std::complex<double>(1.0 / std::cosh(r), 0);

    S[0] = std::sqrt(sechr);
    for (int m = 2; m < N; m += 2) {
        S[m] = utils::SQRT[m - 1] / utils::SQRT[m] * eitheta_tanhr * S[m - 2];
    }
}

void
compute_squeezer(int M, int N, double r, double theta, std::complex<double> *S)
{
    std::complex<double> eitheta_tanhr = std::exp(std::complex<double>(0, theta)) * std::tanh(r);
    std::complex<double> eitheta_tanhr_conj = std::conj(eitheta_tanhr);
    std::complex<double> sechr = std::complex<double>(1.0 / std::cosh(r), 0);

    S[0] = std::sqrt(sechr);

    auto idx = [M, N](int m, int n) -> int {
        if (m < 0) {
            m = M + m;
        }
        if (n < 0) {
            n = N + n;
        }
        return m * N + n;
    };

    for (int m = 2; m < M; m += 2) {
        S[idx(m, 0)] = -utils::SQRT[m - 1] / utils::SQRT[m] * eitheta_tanhr * S[idx(m - 2, 0)];
    }

    for (int m = 0; m < M; m++) {
        for (int n = 2 - (m % 2); n < N; n += 2) {
            if ((m + n) % 2 == 0) {
                std::complex<double> term1 =
                    (utils::SQRT[n - 1] / utils::SQRT[n]) * eitheta_tanhr_conj * S[idx(m, n - 2)];
                std::complex<double> term2 =
                    (utils::SQRT[m] / utils::SQRT[n]) * sechr * S[idx(m - 1, n - 1)];
                S[idx(m, n)] = term1 + term2;
            }
        }
    }
}
}  // namespace strategies
