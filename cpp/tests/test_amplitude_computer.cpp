#include <cstdlib>
#include <vector>

#include <catch2/catch_all.hpp>

#include "AmplitudeComputer.hpp"
#include "test_helper.hpp"

using namespace Mustard;

std::vector<size_t> all_skips_in_level(const AmplitudeComputer& ac, size_t n)
{
	const size_t level_size = ac.level_size(n);
	std::vector<size_t> res;
	res.reserve(level_size);

	for(size_t idx = 0; idx < level_size; idx++)
	{
		res.emplace_back(ac.skips_in_level(n, idx));
	}
	return res;
}

TEST_CASE("Test starting indices", "[AmplitudeComputer]")
{
	auto test_modes = {1, 2, 3, 5, 8, 11, 15};
	for(size_t modes : test_modes)
	{
		auto ac = AmplitudeComputer(modes);

		ac.increase_max_n(10);

		REQUIRE(ac.modes() == modes);

		for(size_t n = 0; n <= 10; n++)
		{
			REQUIRE((ac.starting_indices(n + 1) - ac.starting_indices(n))
					== binomial_coeff(modes + n - 1, modes - 1));
		}
	}
}

TEST_CASE("Test skips", "[AmplitudeComputer]")
{
	SECTION("Modes = 5")
	{
		const size_t modes = 5;
		auto ac = AmplitudeComputer(modes);
		ac.increase_max_n(5);

		REQUIRE(all_skips_in_level(ac, 0) == std::vector<size_t>{0});
		REQUIRE(all_skips_in_level(ac, 1) == std::vector<size_t>{0, 1, 2, 3, 4});
		REQUIRE(all_skips_in_level(ac, 2)
				== std::vector<size_t>{0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4});
		REQUIRE(all_skips_in_level(ac, 3) == std::vector<size_t>{0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3,
																 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
																 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
		REQUIRE(all_skips_in_level(ac, 4)
				== std::vector<size_t>{0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
									   3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4,
									   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
									   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
		REQUIRE(all_skips_in_level(ac, 5)
				== std::vector<size_t>{
					0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
					3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
					3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
					4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
					4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
	}
	SECTION("Modes = 7")
	{
		const size_t modes = 7;
		auto ac = AmplitudeComputer(modes);
		ac.increase_max_n(4);

		REQUIRE(all_skips_in_level(ac, 0) == std::vector<size_t>{0});

		REQUIRE(all_skips_in_level(ac, 1) == std::vector<size_t>{0, 1, 2, 3, 4, 5, 6});
		REQUIRE(all_skips_in_level(ac, 2) == std::vector<size_t>{0, 1, 1, 2, 2, 2, 3, 3, 3, 3,
																 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
																 5, 6, 6, 6, 6, 6, 6, 6});
		REQUIRE(all_skips_in_level(ac, 3)
				== std::vector<size_t>{0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
									   3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
									   4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
									   5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
									   6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6});

		REQUIRE(all_skips_in_level(ac, 4)
				== std::vector<size_t>{
					0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
					3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
					4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
					5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
					5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6,
					6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
					6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
					6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6});
	}
}

TEST_CASE("Test fill next level", "[AmplitudeComputer]")
{
	using namespace Catch::literals;
	using Mustard::Approx;
	std::complex<double> G_minus_one{0.0, 0.0};

	SECTION("D(0.8 + 1.2j) * 5")
	{
		const size_t modes = 5;
		std::vector<std::complex<double>> A(modes * modes, {0.0, 0.0});
		std::vector<std::complex<double>> b(modes, {0.8, -1.2});

		std::vector<std::complex<double>> G0{{0.0055165644207607716, 0.0}};
		std::vector<std::complex<double>> G1_expected(5, {0.00441325, -0.00661988});
		std::vector<std::complex<double>> G2_expected{
			{-0.00312064, -0.00748954}, {-0.00441325, -0.0105918}, {-0.00312064, -0.00748954},
			{-0.00441325, -0.0105918},	{-0.00441325, -0.0105918}, {-0.00312064, -0.00748954},
			{-0.00441325, -0.0105918},	{-0.00441325, -0.0105918}, {-0.00441325, -0.0105918},
			{-0.00312064, -0.00748954}, {-0.00441325, -0.0105918}, {-0.00441325, -0.0105918},
			{-0.00441325, -0.0105918},	{-0.00441325, -0.0105918}, {-0.00312064, -0.00748954}};

		std::vector<std::complex<double>> G3_expected{
			{-0.00663026, -0.00129723}, {-0.01148396, -0.00224686}, {-0.01148396, -0.00224686},
			{-0.00663026, -0.00129723}, {-0.01148396, -0.00224686}, {-0.01624077, -0.00317754},
			{-0.01148396, -0.00224686}, {-0.01148396, -0.00224686}, {-0.01148396, -0.00224686},
			{-0.00663026, -0.00129723}, {-0.01148396, -0.00224686}, {-0.01624077, -0.00317754},
			{-0.01148396, -0.00224686}, {-0.01624077, -0.00317754}, {-0.01624077, -0.00317754},
			{-0.01148396, -0.00224686}, {-0.01148396, -0.00224686}, {-0.01148396, -0.00224686},
			{-0.01148396, -0.00224686}, {-0.00663026, -0.00129723}, {-0.01148396, -0.00224686},
			{-0.01624077, -0.00317754}, {-0.01148396, -0.00224686}, {-0.01624077, -0.00317754},
			{-0.01624077, -0.00317754}, {-0.01148396, -0.00224686}, {-0.01624077, -0.00317754},
			{-0.01624077, -0.00317754}, {-0.01624077, -0.00317754}, {-0.01148396, -0.00224686},
			{-0.01148396, -0.00224686}, {-0.01148396, -0.00224686}, {-0.01148396, -0.00224686},
			{-0.01148396, -0.00224686}, {-0.00663026, -0.00129723}};

		auto ac = AmplitudeComputer(modes);
		std::vector<std::complex<double>> G1(ac.level_size(1));
		std::vector<std::complex<double>> G2(ac.level_size(2));
		std::vector<std::complex<double>> G3(ac.level_size(3));

		double norm1 = ac.fill_next_level(A, b, std::span{&G_minus_one, 1}, G0, G1, 0);
		double norm2 = ac.fill_next_level(A, b, G0, G1, G2, 1);
		double norm3 = ac.fill_next_level(A, b, G1, G2, G3, 2);

		REQUIRE_THAT(G1, Approx(G1_expected).margin(1e-5));
		REQUIRE_THAT(G2, Approx(G2_expected).margin(1e-5));
		REQUIRE_THAT(G3, Approx(G3_expected).margin(1e-5));

		REQUIRE(norm1 == 0.00031649782_a);
		REQUIRE(norm2 == 0.00164578868_a);
		REQUIRE(norm3 == 0.00570540076_a);
	}
}
