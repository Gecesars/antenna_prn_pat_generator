#pragma once

#include <string>

#include "eftx/pattern.hpp"

namespace eftx {

Pattern parse_hfss_csv(const std::string& path);
Pattern parse_generic_table(const std::string& path);
Pattern parse_auto(const std::string& path);

} // namespace eftx

