#include "utils.hpp"
namespace Mustard {
BigUInt pivot_to_rep(const std::vector<size_t>& pivot) {
    if(pivot.size() == 0) {
        return 0;
    }
    BigUInt res = 0;
    for(size_t n = 0; n < pivot.size() - 1; n++) {
        res <<= pivot[n];
        res |= fill_ones(pivot[n]);
        res <<= 1;
    }

    // n == 0
    const auto last = pivot.back();
    res <<= last;
    res |= fill_ones(last);
    return res;
}

std::vector<size_t> rep_to_pivot(size_t modes, BigUInt val) {
    std::vector<size_t> res;
    res.reserve(modes);

    while(val != 0) {
        if((val & 1) == 0) {
            val >>= 1;
            res.emplace_back(0);
        } else {
            size_t cnt = count_trailing_ones(val);
            val >>= (cnt + 1);
            res.push_back(cnt);
        }
    }

    res.resize(modes);

    std::reverse(res.begin(), res.end());
    return res;
}
} // namespace Mustard
