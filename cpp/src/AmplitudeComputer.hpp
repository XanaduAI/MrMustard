#pragma once
#include "utils.hpp"

#include <cstdlib>
#include <span>

namespace Mustard {
class AmplitudeComputer {
private:
    size_t modes_;
    size_t max_n_; // maximum photon number we handle (include)

    std::vector<size_t> starting_indices_;
    std::vector<BigUInt> pivot_reps_;
    std::vector<size_t> skips_;

public:
    AmplitudeComputer(size_t modes) : modes_{modes} {
        max_n_ = 0;

        starting_indices_.push_back(0);
        starting_indices_.push_back(1);

        pivot_reps_.push_back(0);
        skips_.push_back(0);
    }

    size_t level_size(size_t n) const { return binomial_coeff(modes_ + n - 1, modes_ - 1); }

    void increase_max_n(size_t new_max_n) {
        assert(new_max_n > max_n_);

        starting_indices_.resize(new_max_n + 2);

        for(size_t n = max_n_ + 2; n <= new_max_n + 1; n++) {
            starting_indices_[n] = starting_indices_[n - 1] + level_size(n - 1);
        }

        pivot_reps_.reserve(starting_indices_[new_max_n + 1]);

        for(size_t n = max_n_ + 1; n <= new_max_n; n++) {
            MultisetGenerator mgntr(modes_, n);
            for(auto iter = mgntr.begin(); iter != mgntr.end(); ++iter) {
                pivot_reps_.push_back(*iter);
            }
        }

        skips_.reserve(starting_indices_[new_max_n + 1]);
        for(size_t n = max_n_ + 1; n <= new_max_n; n++) {
            for(size_t m = 0; m < modes_; m++) {
                skips_.insert(skips_.end(), binomial_coeff(m + n - 1, m), m);
            }
        }

        max_n_ = new_max_n;
    }

    size_t get_rep_idx_in_level(size_t n, BigUInt rep) const {
        assert(n == count_ones(rep));

        const auto level_start_iter = pivot_reps_.begin() + starting_indices_[n];
        const auto level_end_iter = pivot_reps_.begin() + starting_indices_[n + 1];
        const auto iter = std::find(level_start_iter, level_end_iter, rep);
        return std::distance(level_start_iter, iter);
    }

    std::vector<size_t> upper_indices_in_level(size_t n, BigUInt rep, size_t skip) const {
        assert(n == count_ones(rep));
        MultisetGenerator mgntr(modes_, n);

        std::vector<size_t> indices;
        auto pivot = mgntr.to_pivot(rep);

        for(size_t idx = 0; idx < pivot.size() - skip; idx++) {
            ++pivot[idx];
            const auto rep_up = pivot_to_rep(pivot);
            indices.push_back(get_rep_idx_in_level(n + 1, rep_up));
            --pivot[idx];
        }
        return indices;
    }

    std::vector<std::pair<size_t, size_t>> lower_indices_in_level(size_t n, BigUInt rep) const {
        assert(n == count_ones(rep));
        MultisetGenerator mgntr(modes_, n);

        std::vector<std::pair<size_t, size_t>> indices;
        auto pivot = mgntr.to_pivot(rep);

        for(size_t idx = 0; idx < pivot.size(); idx++) {
            if(pivot[idx] > 0) {
                --pivot[idx];
                const auto rep_low = pivot_to_rep(pivot);
                indices.emplace_back(idx, get_rep_idx_in_level(n - 1, rep_low));
                ++pivot[idx];
            }
        }
        return indices;
    }

    size_t modes() const { return modes_; }

    size_t max_n() const { return max_n_; }

    size_t starting_indices(size_t n) const {
        assert(n <= max_n_ + 1);
        return starting_indices_[n];
    }

    size_t skips(size_t pivot_idx) const {
        assert(pivot_idx < skips_.size());
        return skips_[pivot_idx];
    }

    size_t skips_in_level(size_t n, size_t level_idx) const {
        assert(n <= max_n_);
        return skips_[starting_indices_[n] + level_idx];
    }

    double consume_one_pivot(const std::span<const std::complex<double>>& A,
                             const std::span<const std::complex<double>>& b,
                             const std::span<const std::complex<double>>& G_lower_level,
                             const std::span<const std::complex<double>>& G_curr_level,
                             std::span<std::complex<double>> G_upper_level, size_t n,
                             size_t level_idx) const { // TODO: add restrict
        assert(n + 1 <= max_n_);

        const size_t starting_idx = starting_indices_[n];

        const auto pivot_idx = starting_idx + level_idx;
        const auto rep = pivot_reps_[pivot_idx];

        assert(n == count_ones(rep));

        const auto skip = skips_[pivot_idx];
        const auto pivot = rep_to_pivot(modes_, rep);

        const auto up = upper_indices_in_level(n, rep, skip);

        if(n == 0) { // Do not use lower level
            double norm_sqr = 0.0;
            for(size_t up_idx = 0; up_idx < up.size(); up_idx++) {
                std::complex<double> amplitude = 0.0;

                amplitude += b[up_idx] * G_curr_level[level_idx];
                amplitude /= sqrt(pivot[up_idx] + 1);
                G_upper_level[up[up_idx]] = amplitude;
                norm_sqr += std::norm(amplitude); // squared_norm
            }
            return norm_sqr;
        }

        const auto lo = lower_indices_in_level(n, rep);

        double norm_sqr = 0.0;

        // TODO: add OpenMP
        for(size_t up_idx = 0; up_idx < up.size(); up_idx++) {
            std::complex<double> amplitude = 0.0;

            amplitude += b[up_idx] * G_curr_level[level_idx];
            for(const auto [m, lower_idx] : lo) {
                amplitude += sqrt(pivot[m]) * A[up_idx * modes_ + m] * G_lower_level[lower_idx];
            }
            amplitude /= sqrt(pivot[up_idx] + 1);
            G_upper_level[up[up_idx]] = amplitude;
            norm_sqr += std::norm(amplitude); // squared_norm
        }
        return norm_sqr;
    }

    double fill_next_level(const std::span<const std::complex<double>>& A,
                           const std::span<const std::complex<double>>& b,
                           const std::span<const std::complex<double>>& G_lower_level,
                           const std::span<const std::complex<double>>& G_curr_level,
                           std::span<std::complex<double>> G_upper_level,
                           size_t curr_level) { // TODO: add restrict
        assert(A.size() == modes_ * modes_);
        assert(b.size() == modes_);
        assert(G_lower_level.size() == (curr_level == 0 ? 1 : level_size(curr_level - 1)));
        assert(G_curr_level.size() == level_size(curr_level));
        assert(G_upper_level.size() == level_size(curr_level + 1));

        if(max_n_ < curr_level + 1) {
            increase_max_n(curr_level + 1);
        }

        double norm_sqr = 0.0;
        for(size_t level_idx = 0; level_idx < level_size(curr_level); level_idx++) {
            norm_sqr += consume_one_pivot(A, b, G_lower_level, G_curr_level, G_upper_level,
                                          curr_level, level_idx);
        }
        return norm_sqr;
    }
};
} // namespace Mustard
