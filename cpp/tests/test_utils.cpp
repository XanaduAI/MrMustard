#include "utils.hpp"

#include <catch2/catch_all.hpp>

#include <iostream>

using namespace Mustard;

TEST_CASE("Binomial coefficients", "[utils]") {
	SECTION("Coefficients are correct") {
		auto bc1 = BinomialCoeffs(10);
		for(size_t n = 0; n < 10; n++) {
			for(size_t k = 0; k <= n; k++) {
				REQUIRE(bc1.coeffs(n, k) == binomial_coeff(n, k));
			}
		}
	}

	SECTION("Coefficient must not depend on max_n") {
		auto bc1 = BinomialCoeffs(10);
		auto bc2 = BinomialCoeffs(20);

		REQUIRE(bc1.coeffs(5, 0) == bc2.coeffs(5, 0));
		REQUIRE(bc1.coeffs(5, 3) == bc2.coeffs(5, 3));
		REQUIRE(bc1.coeffs(5, 5) == bc2.coeffs(5, 5));

		REQUIRE(bc1.coeffs(7, 5) == bc2.coeffs(7, 5));
		REQUIRE(bc1.coeffs(7, 7) == bc2.coeffs(7, 7));

		REQUIRE(bc1.coeffs(10, 5) == bc2.coeffs(10, 5));
	}
}

TEST_CASE("Pivots are correct") {
	SECTION("modes = 3, photons = 2") {
		MultisetGenerator mgntr(3, 2);

		std::vector<BigUInt> reps;
		reps.reserve(mgntr.size());
		for(auto iter = mgntr.begin(); iter != mgntr.end(); ++iter) {
			reps.emplace_back(*iter);
		}

		REQUIRE(reps.size() == 6);

		REQUIRE(reps[0] == 0b0011);
		REQUIRE(reps[1] == 0b0101);
		REQUIRE(reps[2] == 0b0110);
		REQUIRE(reps[3] == 0b1001);
		REQUIRE(reps[4] == 0b1010);
		REQUIRE(reps[5] == 0b1100);

		REQUIRE(mgntr.to_pivot(reps[0]) == std::vector<size_t>({0, 0, 2}));
		REQUIRE(mgntr.to_pivot(reps[1]) == std::vector<size_t>({0, 1, 1}));
		REQUIRE(mgntr.to_pivot(reps[2]) == std::vector<size_t>({0, 2, 0}));
		REQUIRE(mgntr.to_pivot(reps[3]) == std::vector<size_t>({1, 0, 1}));
		REQUIRE(mgntr.to_pivot(reps[4]) == std::vector<size_t>({1, 1, 0}));
		REQUIRE(mgntr.to_pivot(reps[5]) == std::vector<size_t>({2, 0, 0}));
	}

	SECTION("modes = 4, photons = 1") {
		MultisetGenerator mgntr(4, 1);

		std::vector<BigUInt> reps;
		reps.reserve(mgntr.size());
		for(auto iter = mgntr.begin(); iter != mgntr.end(); ++iter) {
			reps.emplace_back(*iter);
		}

		REQUIRE(reps.size() == 4);

		REQUIRE(reps[0] == 0b0001);
		REQUIRE(reps[1] == 0b0010);
		REQUIRE(reps[2] == 0b0100);
		REQUIRE(reps[3] == 0b1000);

		REQUIRE(mgntr.to_pivot(reps[0]) == std::vector<size_t>({0, 0, 0, 1}));
		REQUIRE(mgntr.to_pivot(reps[1]) == std::vector<size_t>({0, 0, 1, 0}));
		REQUIRE(mgntr.to_pivot(reps[2]) == std::vector<size_t>({0, 1, 0, 0}));
		REQUIRE(mgntr.to_pivot(reps[3]) == std::vector<size_t>({1, 0, 0, 0}));
	}

	SECTION("modes = 4, photons = 2") {
		MultisetGenerator mgntr(4, 2);

		std::vector<BigUInt> reps;
		reps.reserve(mgntr.size());
		for(auto iter = mgntr.begin(); iter != mgntr.end(); ++iter) {
			reps.emplace_back(*iter);
		}

		REQUIRE(reps.size() == 10);

		REQUIRE(reps[0] == 0b00011);
		REQUIRE(reps[1] == 0b00101);
		REQUIRE(reps[2] == 0b00110);
		REQUIRE(reps[3] == 0b01001);
		REQUIRE(reps[4] == 0b01010);
		REQUIRE(reps[5] == 0b01100);
		REQUIRE(reps[6] == 0b10001);
		REQUIRE(reps[7] == 0b10010);
		REQUIRE(reps[8] == 0b10100);
		REQUIRE(reps[9] == 0b11000);

		REQUIRE(mgntr.to_pivot(reps[0]) == std::vector<size_t>({0, 0, 0, 2}));
		REQUIRE(mgntr.to_pivot(reps[1]) == std::vector<size_t>({0, 0, 1, 1}));
		REQUIRE(mgntr.to_pivot(reps[2]) == std::vector<size_t>({0, 0, 2, 0}));
		REQUIRE(mgntr.to_pivot(reps[3]) == std::vector<size_t>({0, 1, 0, 1}));
		REQUIRE(mgntr.to_pivot(reps[4]) == std::vector<size_t>({0, 1, 1, 0}));
		REQUIRE(mgntr.to_pivot(reps[5]) == std::vector<size_t>({0, 2, 0, 0}));
		REQUIRE(mgntr.to_pivot(reps[6]) == std::vector<size_t>({1, 0, 0, 1}));
		REQUIRE(mgntr.to_pivot(reps[7]) == std::vector<size_t>({1, 0, 1, 0}));
		REQUIRE(mgntr.to_pivot(reps[8]) == std::vector<size_t>({1, 1, 0, 0}));
		REQUIRE(mgntr.to_pivot(reps[9]) == std::vector<size_t>({2, 0, 0, 0}));
	}

	SECTION("modes = 4, photons = 3") {
		MultisetGenerator mgntr(4, 3);

		std::vector<BigUInt> reps;
		reps.reserve(mgntr.size());
		for(auto iter = mgntr.begin(); iter != mgntr.end(); ++iter) {
			reps.emplace_back(*iter);
		}

		REQUIRE(reps.size() == 20);

		REQUIRE(mgntr.to_pivot(reps[0]) == std::vector<size_t>({0, 0, 0, 3}));
		REQUIRE(mgntr.to_pivot(reps[1]) == std::vector<size_t>({0, 0, 1, 2}));
		REQUIRE(mgntr.to_pivot(reps[2]) == std::vector<size_t>({0, 0, 2, 1}));
		REQUIRE(mgntr.to_pivot(reps[3]) == std::vector<size_t>({0, 0, 3, 0}));
		REQUIRE(mgntr.to_pivot(reps[4]) == std::vector<size_t>({0, 1, 0, 2}));
		REQUIRE(mgntr.to_pivot(reps[5]) == std::vector<size_t>({0, 1, 1, 1}));
		REQUIRE(mgntr.to_pivot(reps[6]) == std::vector<size_t>({0, 1, 2, 0}));
		REQUIRE(mgntr.to_pivot(reps[7]) == std::vector<size_t>({0, 2, 0, 1}));
		REQUIRE(mgntr.to_pivot(reps[8]) == std::vector<size_t>({0, 2, 1, 0}));
		REQUIRE(mgntr.to_pivot(reps[9]) == std::vector<size_t>({0, 3, 0, 0}));
		REQUIRE(mgntr.to_pivot(reps[10]) == std::vector<size_t>({1, 0, 0, 2}));
		REQUIRE(mgntr.to_pivot(reps[11]) == std::vector<size_t>({1, 0, 1, 1}));
		REQUIRE(mgntr.to_pivot(reps[12]) == std::vector<size_t>({1, 0, 2, 0}));
		REQUIRE(mgntr.to_pivot(reps[13]) == std::vector<size_t>({1, 1, 0, 1}));
		REQUIRE(mgntr.to_pivot(reps[14]) == std::vector<size_t>({1, 1, 1, 0}));
		REQUIRE(mgntr.to_pivot(reps[15]) == std::vector<size_t>({1, 2, 0, 0}));
		REQUIRE(mgntr.to_pivot(reps[16]) == std::vector<size_t>({2, 0, 0, 1}));
		REQUIRE(mgntr.to_pivot(reps[17]) == std::vector<size_t>({2, 0, 1, 0}));
		REQUIRE(mgntr.to_pivot(reps[18]) == std::vector<size_t>({2, 1, 0, 0}));
		REQUIRE(mgntr.to_pivot(reps[19]) == std::vector<size_t>({3, 0, 0, 0}));
	}

	SECTION("Test to representation") {
		for(uint32_t modes = 1; modes <= 10; modes++) {
			for(uint32_t photons = 0; photons < 10; photons++) {
				MultisetGenerator mgntr(modes, photons);

				std::vector<BigUInt> reps;
				reps.reserve(mgntr.size());
				for(auto iter = mgntr.begin(); iter != mgntr.end(); ++iter) {
					reps.emplace_back(*iter);
				}

				for(const auto& r : reps) {
					REQUIRE(pivot_to_rep(mgntr.to_pivot(r)) == r);
				}
			}
		}
	}
}
