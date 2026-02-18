#pragma once

#include <string>

#include "eftx/pattern.hpp"

namespace eftx {

Pattern normalize_pattern(const Pattern& in, const std::string& mode);
Pattern resample_vertical(const Pattern& in, const std::string& norm = "none");
Pattern resample_vertical_adt(const Pattern& in);
Pattern resample_horizontal(const Pattern& in, const std::string& norm = "none");

} // namespace eftx

