#pragma once

#include <string>

#include "eftx/pattern.hpp"

namespace eftx {

struct ProjectData {
    std::string base_name;

    Pattern h1;
    Pattern v1;
    Pattern h2;
    Pattern v2;

    bool has_h1 = false;
    bool has_v1 = false;
    bool has_h2 = false;
    bool has_v2 = false;
};

ProjectData load_project_json(const std::string& path);

} // namespace eftx
