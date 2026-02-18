#pragma once

#include <string>
#include <vector>

#include "eftx/pattern.hpp"

namespace eftx {

struct CutMetrics {
    double peak_lin = 0.0;
    double peak_db = 0.0;
    double min_db = 0.0;
    double hpbw_deg = 0.0;
    double d2d_lin = 0.0;
    double d2d_db = 0.0;
};

struct DashboardCut {
    std::string title;
    PatternKind kind = PatternKind::Unknown;
    Pattern pattern;
    std::string image_relpath;
    std::string table_relpath;
    std::string pat_relpath;
    std::string adt_relpath;
};

CutMetrics compute_cut_metrics(const Pattern& cut, PatternKind kind);

void write_cut_svg(
    const std::string& path,
    const Pattern& cut,
    PatternKind kind,
    const std::string& title
);

void write_project_dashboard(
    const std::string& html_path,
    const std::string& project_name,
    const std::vector<DashboardCut>& cuts,
    const std::vector<std::string>& extra_artifacts
);

} // namespace eftx

