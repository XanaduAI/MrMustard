#pragma once
#include "utils.hpp"
#include "macros.hpp"

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

    mutable BinomialCoeffs binom_;

public:
    AmplitudeComputer(size_t modes) : modes_{modes}, binom_{10} {
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

        binom_.resize(new_max_n + modes_);

        skips_.reserve(starting_indices_[new_max_n + 1]);
        for(size_t n = max_n_ + 1; n <= new_max_n; n++) {
            for(size_t m = 0; m < modes_; m++) {
                skips_.insert(skips_.end(), binom_.coeff(m + n - 1, m), m);
            }
        }

        max_n_ = new_max_n;
    }

    size_t get_rep_idx_in_level(size_t n, BigUInt rep) const {
        assert(n == static_cast<size_t>(count_ones(rep)));

        const auto level_start_iter = pivot_reps_.begin() + starting_indices_[n];
        const auto level_end_iter = pivot_reps_.begin() + starting_indices_[n + 1];
        const auto iter = std::lower_bound(level_start_iter, level_end_iter, rep);
        return std::distance(level_start_iter, iter);
    }

    size_t get_pivot_idx_in_level(size_t n, const std::vector<size_t>& pivot) const {
        assert(n == std::accumulate(pivot.begin(), pivot.end(), size_t{0}));

        size_t idx = 0;
        size_t modes_l = modes_;
        for(size_t m = 0; m < modes_l; m++) {
            size_t t = pivot[m];
            idx += binom_.coeff(modes_l - m + n - 1, n) - binom_.coeff(modes_l - m + n - t - 1, n-t);
            n -= t;
        }
        return idx;
    }

    std::vector<size_t> upper_indices_in_level(size_t n, BigUInt rep, size_t skip) const {
        assert(n == static_cast<size_t>(count_ones(rep)));

        std::vector<size_t> indices;
        indices.reserve(modes_);
        auto pivot = rep_to_pivot(modes_, rep);

        for(size_t idx = 0; idx < pivot.size() - skip; idx++) {
            ++pivot[idx];
            indices.push_back(get_pivot_idx_in_level(n+1, pivot));
            --pivot[idx];
        }
        return indices;
    }

    std::vector<size_t> lower_indices_in_level(const size_t n, const BigUInt rep) const {
        assert(n == static_cast<size_t>(count_ones(rep)));

        std::vector<size_t> indices(modes_, 0);
        auto pivot = rep_to_pivot(modes_, rep);
        for(size_t idx = 0; idx < pivot.size(); idx++) {
            if(pivot[idx] > 0) {
                --pivot[idx];
                indices[idx] = get_pivot_idx_in_level(n-1, pivot);
                ++pivot[idx];
            }
        }
        return indices;
    }

    /*
    std::vector<size_t> lower_indices_in_level(const size_t n, const BigUInt rep) const {
        assert(n == static_cast<size_t>(count_ones(rep)));

        size_t pos = 0;
        std::vector<BigUInt> first_one_pos;
        first_one_pos.reserve(modes_);

        size_t s = rep;
        while(s) {
            if (s & 1) {
                first_one_pos.emplace_back(pos);
                pos += count_trailing_ones(s)+1;
                s = (rep >> pos);
            } else {
                s >>= 1;
                ++pos;
                first_one_pos.emplace_back(~size_t{0});
            }
        }

        std::vector<size_t> indices(modes_, 0);
        for(size_t m = 0; m < first_one_pos.size(); m++) {
            const auto pos = first_one_pos[m];
            if(pos == ~size_t{0}) {
                continue;
            }
            const auto rep_low = ((rep >> 1) & upper_parity(pos)) | (rep & lower_parity(pos));
            indices[modes_ - m - 1] = get_rep_idx_in_level(n - 1, rep_low);
        }
        
        return indices;
    }
    */

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

    double consume_one_pivot(const std::complex<double>* A, const std::complex<double>* b,
                             const std::complex<double>* G_lower_level,
                             const std::complex<double>* G_curr_level,
                             std::complex<double>* G_upper_level, size_t n,
                             size_t level_idx) const { // TODO: add restrict
        assert(n + 1 <= max_n_);

        const size_t starting_idx = starting_indices_[n];

        const auto pivot_idx = starting_idx + level_idx;
        const auto rep = pivot_reps_[pivot_idx];

        assert(n == static_cast<size_t>(count_ones(rep)));

        const auto skip = skips_[pivot_idx];
        const auto pivot = rep_to_pivot(modes_, rep);

        const auto up = upper_indices_in_level(n, rep, skip);

        if(n == 0) { // Do not use lower level
            double norm_sqr = 0.0;
            for(size_t up_idx = 0; up_idx < up.size(); up_idx++) {
                std::complex<double> amplitude = b[up_idx] * G_curr_level[level_idx];
                amplitude /= sqrt(pivot[up_idx] + 1);
                G_upper_level[up[up_idx]] = amplitude;
                norm_sqr += std::norm(amplitude); // squared_norm
            }
            return norm_sqr;
        }

        const auto lo = lower_indices_in_level(n, rep);

        double norm_sqr = 0.0;
        const auto modes_l = modes_; // to stack for performance

        std::vector<std::complex<double>> G_lower_tmp(lo.size());

        for(size_t m = 0; m < modes_l; m++) {
            // We have lo[m] == 0 if pivot[m] == 0. Still, as we multiply sqrt(pivot[m]),
            // this does not contribute to amplitude
            G_lower_tmp[m] = sqrt(pivot[m]) * G_lower_level[lo[m]];
        }

        for(size_t up_idx = 0; up_idx < up.size(); up_idx++) {
            std::complex<double> amplitude = 0.0;

            amplitude += b[up_idx] * G_curr_level[level_idx];
            for(size_t m = 0; m < modes_l; m++) {
                amplitude += A[up_idx * modes_l + m] * G_lower_tmp[m];
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

        binom_.resize(curr_level + 1 + modes_);

        double norm_sqr = 0.0;
        const auto curr_level_size = level_size(curr_level);
        for(size_t level_idx = 0; level_idx < curr_level_size; level_idx++) {
            norm_sqr
                += consume_one_pivot(A.data(), b.data(), G_lower_level.data(), G_curr_level.data(),
                                     G_upper_level.data(), curr_level, level_idx);
        }
        return norm_sqr;
    }
};
} // namespace Mustard
