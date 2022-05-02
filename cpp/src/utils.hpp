#include <boost/dynamic_bitset.hpp>

#include <cstdlib>
#include <iterator>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace Mustard
{
using BigUInt = uint64_t;

constexpr size_t binomial_coeff(size_t n, size_t k)
{
	size_t res = 1;

	if(k > n - k)
	{
		k = n - k;
	}

	for(size_t i = 0; i < k; i++)
	{
		res *= (n - i);
		res /= (i + 1);
	}

	return res;
}

class BinomialCoeffs
{
private:
	// May change to one dim vector for cache optimization
	std::vector<std::vector<size_t>> coeffs_;

public:
	explicit BinomialCoeffs(size_t max_n) : coeffs_(max_n + 1)
	{
		for(size_t n = 0; n <= max_n; n++)
		{
			coeffs_[n].resize(n + 1);
			coeffs_[n][0] = 1;
			coeffs_[n][n] = 1;
		}
		for(size_t n = 1; n <= max_n; n++)
		{
			for(size_t k = 1; k < n; k++)
			{
				coeffs_[n][k] = coeffs_[n - 1][k - 1] + coeffs_[n - 1][k];
			}
		}
	}

	size_t coeffs(size_t n, size_t k) const
	{
		assert(k <= n);
		assert(n < coeffs_.size());
		return coeffs_[n][k];
	}
};

constexpr auto count_trailing_zeros(uint64_t n) -> int
{
	return std::countr_zero(n);
}

constexpr auto count_trailing_ones(uint64_t n) -> int
{
	return std::countr_one(n);
}

constexpr auto count_ones(uint64_t n) -> int
{
	return std::popcount(n);
}

constexpr auto fill_ones(uint32_t n) -> BigUInt
{
	return (BigUInt(1U) << n) - 1;
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

BigUInt pivot_to_rep(const std::vector<size_t>& pivot);
std::vector<size_t> rep_to_pivot(size_t modes, BigUInt val);

class MultisetGenerator
{
private:
	// using BigUInt = unsigned __int128;

	size_t m_;
	size_t n_;

public:
	/**
	 * @brief
	 *
	 * We use a binary representation of pivot to save it internally.
	 * In this representation, 0 means a wall and 1 means the element.
	 * E.g. 0110 == [0,2,0]
	 */
	struct RepIterator
	{
		using iterator_category = std::forward_iterator_tag;
		using difference_type = BigUInt;
		using vaule_type = BigUInt;

		BigUInt val_;

		explicit RepIterator(BigUInt val) : val_{val} { }

		RepIterator& operator++() // prefix
		{
			next();
			return *this;
		}
		RepIterator operator++(int) // postfix
		{
			RepIterator r(*this);
			next();
			return r;
		}

		void next()
		{
			BigUInt t = val_ | (val_ - 1);
			BigUInt w = (t + 1) | (((~t & -~t) - 1) >> (count_trailing_zeros(val_) + 1));
			val_ = w;
		}

		auto operator*() const -> BigUInt { return val_; }

		auto operator==(const RepIterator& rhs) const -> bool { return val_ == rhs.val_; }
		auto operator!=(const RepIterator& rhs) const -> bool { return val_ != rhs.val_; }
	};

	/**
	 * @brief Create generator for dividing n elements to m bins.
	 */
	MultisetGenerator(size_t m, size_t n) : m_{m}, n_{n} { assert(m > 0); }

	RepIterator begin() const { return RepIterator{(BigUInt(1) << BigUInt(n_)) - 1}; }

	RepIterator end() const
	{
		RepIterator r{((BigUInt(1) << BigUInt(n_)) - 1) << (m_ - 1)};
		r.next();
		return r;
	}

	size_t size() const { return binomial_coeff(n_ + m_ - 1, n_); }

	std::vector<size_t> to_pivot(BigUInt val) const
	{
		assert(count_ones(val) == n_);

		return Mustard::rep_to_pivot(m_, val);
	}
};

} // namespace Mustard
