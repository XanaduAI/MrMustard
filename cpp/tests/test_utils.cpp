#include "utils.hpp"

#include <catch2/catch.hpp>

using namespace Mustard;

TEST_CASE("Test binomial coefficients", "[utils]") {
    SECTION("Coefficients are correct") {
        auto bc1 = BinomialCoeffs(10);
        for(size_t n = 0; n < 10; n++) { 
            for(size_t k = 0; k <= n; k++) {
                bc1.coeffs(n, k) == binomial_coeff(n, k);
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
