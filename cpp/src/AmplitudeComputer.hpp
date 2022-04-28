#pragma once
namespace Mustard {
class AmplitudeComputer {
  private:
    using LevelT = std::pair<std::size_t, size_t>;
    using PivotIdx = size_t;

    BinomialCoeffs binom_;

  public:
    AmplitudeComputer(size_t modes, size_t max_photons)
        : binom_{modes + max_photons - 1} {
    }

    size_t level_size(size_t modes, size_t photons) const {
        return binom_.coeffs(modes + photons - 1, photons);
    }

    size_t pivot_idx_in_level(const std::vector<size_t>& pivot) {
    }
};
} // namespace Mustard
