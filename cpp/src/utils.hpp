#include <boost/dynamic_bitset.hpp>

#include <iterator>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <numeric>

namespace Mustard {
using BigUInt = uint64_t;

size_t binomial_coeff(size_t n, size_t k) {
    size_t res = 1;

    if(k > n - k) {
        k = n - k;
    }

    for(size_t i = 0; i < k; i++) {
        res *= (n-i);
        res /= (i + 1);
    }

    return res;
}

class BinomialCoeffs {
  private:
    // May change to one dim vector for cache optimization
    std::vector<std::vector<size_t>> coeffs_;

  public:
    explicit BinomialCoeffs(size_t max_n)
        : coeffs_(max_n+1) {
        for(size_t n = 0; n <= max_n; n++) {
            coeffs_[n].resize(n+1);
            coeffs_[n][0] = 1;
            coeffs_[n][n] = 1;
        }
        for(size_t n = 1; n <= max_n; n++) {
            for(size_t k = 1; k < n; k++) {
                coeffs_[n][k] = coeffs_[n-1][k-1] + coeffs_[n-1][k];
            }
        }
    }

    size_t coeffs(size_t n, size_t k) const {
        assert(k <= n);
        assert(n < coeffs_.size());
        return coeffs_[n][k];
    }
};

constexpr auto count_trailing_zeros(uint64_t n) -> int {
    return std::countl_zero(n);
}

constexpr auto count_ones(uint64_t n) -> int {
    return std::popcount(n);
}

/*
constexpr auto count_trailing_zeros(unsigned __int128 n) -> int {
    // NOLINTNEXTLINE (readability-magic-numbers)
    constexpr uint64_t ones_64 = ~uint64_t(0);
    if (const auto mod = (n & ones_64); mod != 0) {
        return count_trailing_zeros(static_cast<uint64_t>(mod));
    }
    // NOLINTNEXTLINE (readability-magic-numbers)
    return count_trailing_zeros(static_cast<uint64_t>(n >> 64U)) + 64U;
}
*/

class MultisetGenerator {
  private:
    //using BigUInt = unsigned __int128;

    size_t m_;
    size_t n_;

  public:
    /**
     * @brief
     *
     * We use a binary representation of pivot to save it internally.
     * In this representation, 0 means an element and 1 means the wall between
     * bins. E.g. 0110 == [1,0,1]
     */
    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = BigUInt;
        using vaule_type = BigUInt;

        const MultisetGenerator& mgntr_;
        BigUInt val_;

        explicit Iterator(const MultisetGenerator& mgntr, BigUInt val)
            : mgntr_{mgntr}, val_{val} {}

        Iterator& operator++() //prefix
        {
            next();
            return *this;
        }
        Iterator operator++(int) //postfix
        {
            Iterator r(*this);
            next();
            return r;
        }

        void next() {
            BigUInt t = val_ | (val_ - 1);
            BigUInt w = (t + 1) | (((~t & -~t) - 1) >> (count_trailing_zeros(val_) + 1));
            val_ = w;
        }

        auto operator*() const -> std::vector<size_t>;

        auto operator==(const Iterator& rhs) -> bool { return val_ == rhs.val_; }
        auto operator!=(const Iterator& rhs) -> bool { return val_ != rhs.val_; }
    };

    /**
     * @brief Create generator for dividing n elements to m bins.
     */
    MultisetGenerator(size_t m, size_t n)
        : m_{m}, n_{n} {
    }

    Iterator begin() const {
        return Iterator{*this, (BigUInt(1) << BigUInt(m_-1)) - 1};
    }

    Iterator end() const {
        return Iterator{*this, (BigUInt(1) << BigUInt(m_-1)) - 1};
    }

    std::vector<size_t> to_pivot(BigUInt val) const {
        assert(count_ones(val) == n_);
        std::vector<size_t> res;

        while(val != 0) {
            if((val & 1) == 1) {
                val >>= 1;
                res.emplace_back(0);
            } else {
                size_t cnt = count_trailing_zeros(val);
                val >>= cnt;
                res.push_back(cnt);
            }
        }

        while (res.size() != m_) {
            res.push_back(0);
        }

        std::reverse(res.begin(), res.end());
        return res;
    }
};

} // namespace Mustard
