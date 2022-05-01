#pragma once

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#include <complex>
#include <type_traits>
#include <sstream>
#include <vector>

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T> &vec) {
    os << '[';
    if (!vec.empty()) {
        for (size_t i = 0; i < vec.size() - 1; i++) {
            os << vec[i] << ", ";
        }
        os << vec.back();
    }
    os << ']';
    return os;
}

namespace Mustard {
template<typename T>
struct is_complex: std::false_type {};

template<typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template<typename T>
struct remove_complex {
    using type = T;
};
template<typename T>
struct remove_complex<std::complex<T>> {
    using type = T;
};
template <typename T>
using remove_complex_type = remove_complex<T>::type;

template<typename T>
constexpr static auto is_complex_v = is_complex<T>::value;

template<typename T, typename AllocComp>
struct ComplexVectorApprox: Catch::Matchers::MatcherGenericBase {
    static_assert(is_complex_v<T>, "Parameter type must be complex.");

  private:
    const std::vector<T>& comp_;
    mutable Catch::Approx approx = Catch::Approx::custom();

  public:
    ComplexVectorApprox(const std::vector<T, AllocComp>& comp)
        : comp_{comp} {
    }

    std::string describe() const {
        std::ostringstream ss;
        ss << "is approx to " << comp_;
        return ss.str();
    }

    template<typename AllocMatch>
    bool match(const std::vector<T, AllocMatch>& v) const {
        if(comp_.size() != v.size()) {
            return false;
        }
        for(size_t i = 0; i < v.size(); i++) {
            if ((std::real(comp_[i]) != approx(std::real(v[i])))
                    ||(std::imag(comp_[i]) != approx(std::imag(v[i])))) {
                return false;
            }
        }
        return true;
    }

    ComplexVectorApprox& epsilon(remove_complex_type<T> new_eps) {
        approx.epsilon(new_eps);
        return *this;
    }

    ComplexVectorApprox& margin(remove_complex_type<T> new_margin) {
        approx.margin(new_margin);
        return *this;
    }
};

template<typename T, typename AllocComp>
ComplexVectorApprox<std::complex<T>, AllocComp> Approx(const std::vector<std::complex<T>, AllocComp>& comp) {
    return ComplexVectorApprox(comp);
}

} // namespace Mustard
