#include "utils.hpp"
namespace Mustard {
auto MultisetGenerator::Iterator::operator*() const -> std::vector<size_t> {
    return mgntr_.to_pivot(val_);
}
} // Mustard
