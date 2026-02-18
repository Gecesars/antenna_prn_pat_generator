#include "eftx/dashboard.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "eftx/metrics.hpp"
#include "eftx/util.hpp"

namespace eftx {

namespace {

static bool is_valid(const Pattern& p) {
    return !p.angles_deg.empty() && p.angles_deg.size() == p.values_lin.size();
}

static void write_text_file(const std::string& path, const std::string& text) {
    std::ofstream ofs(path.c_str(), std::ios::out | std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Could not write file: " + path);
    }
    ofs << text;
}

static std::string html_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 16U);
    for (std::size_t i = 0; i < in.size(); ++i) {
        const char c = in[i];
        if (c == '&') {
            out += "&amp;";
        } else if (c == '<') {
            out += "&lt;";
        } else if (c == '>') {
            out += "&gt;";
        } else if (c == '"') {
            out += "&quot;";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

static std::string web_path(std::string p) {
    std::replace(p.begin(), p.end(), '\\', '/');
    return p;
}

static std::string fmt_num(double v, int prec) {
    if (!std::isfinite(v)) {
        return "-";
    }
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(prec) << v;
    return oss.str();
}

static std::string make_id(const std::string& in, std::size_t idx) {
    std::string out;
    out.reserve(in.size() + 8U);
    for (std::size_t i = 0; i < in.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(in[i]);
        if (std::isalnum(c) != 0) {
            out.push_back(static_cast<char>(std::tolower(c)));
        } else if (c == ' ' || c == '_' || c == '-' || c == '|') {
            out.push_back('-');
        }
    }
    if (out.empty()) {
        out = "cut";
    }
    while (!out.empty() && out.back() == '-') {
        out.pop_back();
    }
    out += "-" + std::to_string(idx + 1U);
    return out;
}

static std::string svg_font() {
    return "Bahnschrift,'Trebuchet MS','Segoe UI',Arial,sans-serif";
}

static std::string svg_metrics_block(double x, double y, const CutMetrics& m) {
    std::ostringstream s;
    s.setf(std::ios::fixed);
    s << std::setprecision(2);
    s << "<g>\n";
    s << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"270\" height=\"112\" rx=\"10\" fill=\"#ffffff\" fill-opacity=\"0.88\" stroke=\"#c9d9ea\"/>\n";
    s << "<text x=\"" << (x + 12.0) << "\" y=\"" << (y + 24.0)
      << "\" font-family=\"" << svg_font() << "\" font-size=\"13\" fill=\"#35506e\">Peak:</text>\n";
    s << "<text x=\"" << (x + 88.0) << "\" y=\"" << (y + 24.0)
      << "\" font-family=\"" << svg_font() << "\" font-size=\"13\" font-weight=\"700\" fill=\"#0f2034\">"
      << fmt_num(m.peak_db, 2) << " dB</text>\n";
    s << "<text x=\"" << (x + 12.0) << "\" y=\"" << (y + 48.0)
      << "\" font-family=\"" << svg_font() << "\" font-size=\"13\" fill=\"#35506e\">HPBW:</text>\n";
    s << "<text x=\"" << (x + 88.0) << "\" y=\"" << (y + 48.0)
      << "\" font-family=\"" << svg_font() << "\" font-size=\"13\" font-weight=\"700\" fill=\"#0f2034\">"
      << fmt_num(m.hpbw_deg, 2) << " deg</text>\n";
    s << "<text x=\"" << (x + 12.0) << "\" y=\"" << (y + 72.0)
      << "\" font-family=\"" << svg_font() << "\" font-size=\"13\" fill=\"#35506e\">D2D:</text>\n";
    s << "<text x=\"" << (x + 88.0) << "\" y=\"" << (y + 72.0)
      << "\" font-family=\"" << svg_font() << "\" font-size=\"13\" font-weight=\"700\" fill=\"#0f2034\">"
      << fmt_num(m.d2d_db, 2) << " dB</text>\n";
    s << "<text x=\"" << (x + 12.0) << "\" y=\"" << (y + 96.0)
      << "\" font-family=\"" << svg_font() << "\" font-size=\"13\" fill=\"#35506e\">Min:</text>\n";
    s << "<text x=\"" << (x + 88.0) << "\" y=\"" << (y + 96.0)
      << "\" font-family=\"" << svg_font() << "\" font-size=\"13\" font-weight=\"700\" fill=\"#0f2034\">"
      << fmt_num(m.min_db, 2) << " dB</text>\n";
    s << "</g>\n";
    return s.str();
}

static std::string render_planar_svg(
    const Pattern& cut,
    const std::string& title
) {
    std::vector<double> ang = cut.angles_deg;
    std::vector<double> val = cut.values_lin;
    sort_pairs(ang, val);

    double vmax = 0.0;
    for (std::size_t i = 0; i < val.size(); ++i) {
        if (val[i] > vmax) {
            vmax = val[i];
        }
    }
    if (vmax <= 0.0) {
        vmax = 1.0;
    }

    const CutMetrics m = compute_cut_metrics(cut, PatternKind::Vertical);
    const double x_min = -90.0;
    const double x_max = 90.0;
    const double db_floor = -40.0;

    const int width = 1280;
    const int height = 560;
    const double left = 92.0;
    const double right = 36.0;
    const double top = 84.0;
    const double bottom = 72.0;
    const double plot_w = static_cast<double>(width) - left - right;
    const double plot_h = static_cast<double>(height) - top - bottom;

    auto x_to_px = [&](double x) {
        const double u = clamp((x - x_min) / (x_max - x_min), 0.0, 1.0);
        return left + u * plot_w;
    };
    auto db_to_py = [&](double db) {
        const double d = clamp((db - db_floor) / (0.0 - db_floor), 0.0, 1.0);
        return top + (1.0 - d) * plot_h;
    };

    std::ostringstream line_pts;
    std::ostringstream fill_pts;
    line_pts.setf(std::ios::fixed);
    fill_pts.setf(std::ios::fixed);
    line_pts << std::setprecision(2);
    fill_pts << std::setprecision(2);

    fill_pts << x_to_px(ang.front()) << "," << db_to_py(db_floor) << " ";
    for (std::size_t i = 0; i < ang.size(); ++i) {
        const double e = std::max(val[i] / vmax, 1e-8);
        const double db = std::max(20.0 * std::log10(e), db_floor);
        const double px = x_to_px(ang[i]);
        const double py = db_to_py(db);
        line_pts << px << "," << py << " ";
        fill_pts << px << "," << py << " ";
    }
    fill_pts << x_to_px(ang.back()) << "," << db_to_py(db_floor);

    std::ostringstream svg;
    svg.setf(std::ios::fixed);
    svg << std::setprecision(2);
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height
        << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    svg << "<defs>\n";
    svg << "<linearGradient id=\"bg\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"1\">\n";
    svg << "<stop offset=\"0%\" stop-color=\"#f6fbff\"/>\n";
    svg << "<stop offset=\"100%\" stop-color=\"#edf5fc\"/>\n";
    svg << "</linearGradient>\n";
    svg << "<linearGradient id=\"area\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"1\">\n";
    svg << "<stop offset=\"0%\" stop-color=\"#0ea5a8\" stop-opacity=\"0.30\"/>\n";
    svg << "<stop offset=\"100%\" stop-color=\"#0ea5a8\" stop-opacity=\"0.04\"/>\n";
    svg << "</linearGradient>\n";
    svg << "</defs>\n";

    svg << "<rect width=\"100%\" height=\"100%\" fill=\"url(#bg)\"/>\n";
    svg << "<rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << plot_w << "\" height=\"" << plot_h
        << "\" fill=\"#ffffff\" stroke=\"#ccdded\" stroke-width=\"1\" rx=\"10\"/>\n";

    for (int db = 0; db >= -40; db -= 5) {
        const bool major = (db % 10) == 0;
        const double y = db_to_py(static_cast<double>(db));
        svg << "<line x1=\"" << left << "\" y1=\"" << y << "\" x2=\"" << (left + plot_w) << "\" y2=\"" << y
            << "\" stroke=\"" << (major ? "#d8e7f4" : "#edf3f9") << "\" stroke-width=\"1\"/>\n";
        if (major) {
            svg << "<text x=\"" << (left - 10.0) << "\" y=\"" << (y + 4.0)
                << "\" text-anchor=\"end\" font-family=\"" << svg_font()
                << "\" font-size=\"12\" fill=\"#4f667f\">" << db << " dB</text>\n";
        }
    }

    for (int a = -90; a <= 90; a += 15) {
        const bool major = (a % 30) == 0;
        const double x = x_to_px(static_cast<double>(a));
        svg << "<line x1=\"" << x << "\" y1=\"" << top << "\" x2=\"" << x << "\" y2=\"" << (top + plot_h)
            << "\" stroke=\"" << (major ? "#e2edf8" : "#f2f6fb") << "\" stroke-width=\"1\"/>\n";
        if (major) {
            svg << "<text x=\"" << x << "\" y=\"" << (top + plot_h + 24.0)
                << "\" text-anchor=\"middle\" font-family=\"" << svg_font()
                << "\" font-size=\"12\" fill=\"#4f667f\">" << a << "</text>\n";
        }
    }

    svg << "<polygon points=\"" << fill_pts.str() << "\" fill=\"url(#area)\"/>\n";
    svg << "<polyline fill=\"none\" stroke=\"#0d9488\" stroke-width=\"2.8\" points=\"" << line_pts.str() << "\"/>\n";
    svg << "<line x1=\"" << x_to_px(0.0) << "\" y1=\"" << top << "\" x2=\"" << x_to_px(0.0) << "\" y2=\"" << (top + plot_h)
        << "\" stroke=\"#6ea2cf\" stroke-width=\"1.2\" stroke-dasharray=\"5,4\"/>\n";

    svg << "<text x=\"" << left << "\" y=\"40\" font-family=\"" << svg_font()
        << "\" font-size=\"28\" font-weight=\"700\" fill=\"#10243c\">" << html_escape(title) << "</text>\n";
    svg << "<text x=\"" << left << "\" y=\"62\" font-family=\"" << svg_font()
        << "\" font-size=\"13\" fill=\"#4f637b\">Elevacao (graus) | Nivel relativo (dB)</text>\n";
    svg << svg_metrics_block(left + plot_w - 282.0, 16.0, m);
    svg << "</svg>\n";
    return svg.str();
}

static std::string render_polar_svg(
    const Pattern& cut,
    const std::string& title
) {
    std::vector<double> ang = cut.angles_deg;
    std::vector<double> val = cut.values_lin;
    sort_pairs(ang, val);

    double vmax = 0.0;
    for (std::size_t i = 0; i < val.size(); ++i) {
        if (val[i] > vmax) {
            vmax = val[i];
        }
    }
    if (vmax <= 0.0) {
        vmax = 1.0;
    }

    const CutMetrics m = compute_cut_metrics(cut, PatternKind::Horizontal);
    const double db_floor = -40.0;

    const int width = 1280;
    const int height = 560;
    const double cx = 380.0;
    const double cy = 320.0;
    const double r_outer = 210.0;
    const double r_inner = 34.0;

    auto db_to_r = [&](double db) {
        const double t = clamp((db - db_floor) / (0.0 - db_floor), 0.0, 1.0);
        return r_inner + t * (r_outer - r_inner);
    };
    auto pt_xy = [&](double a_deg, double r) {
        const double rad = (90.0 - a_deg) * 3.14159265358979323846 / 180.0;
        const double x = cx + r * std::cos(rad);
        const double y = cy - r * std::sin(rad);
        return std::pair<double, double>(x, y);
    };

    std::ostringstream poly;
    poly.setf(std::ios::fixed);
    poly << std::setprecision(2);
    for (std::size_t i = 0; i < ang.size(); ++i) {
        const double e = std::max(val[i] / vmax, 1e-8);
        const double db = std::max(20.0 * std::log10(e), db_floor);
        const double r = db_to_r(db);
        const auto xy = pt_xy(ang[i], r);
        poly << xy.first << "," << xy.second << " ";
    }

    std::ostringstream svg;
    svg.setf(std::ios::fixed);
    svg << std::setprecision(2);
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height
        << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    svg << "<defs>\n";
    svg << "<radialGradient id=\"bg\" cx=\"30%\" cy=\"20%\" r=\"95%\">\n";
    svg << "<stop offset=\"0%\" stop-color=\"#f8fbff\"/>\n";
    svg << "<stop offset=\"100%\" stop-color=\"#edf5fd\"/>\n";
    svg << "</radialGradient>\n";
    svg << "<linearGradient id=\"strokeA\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\">\n";
    svg << "<stop offset=\"0%\" stop-color=\"#0d9488\"/>\n";
    svg << "<stop offset=\"100%\" stop-color=\"#0f4c81\"/>\n";
    svg << "</linearGradient>\n";
    svg << "</defs>\n";

    svg << "<rect width=\"100%\" height=\"100%\" fill=\"url(#bg)\"/>\n";
    svg << "<rect x=\"34\" y=\"82\" width=\"748\" height=\"450\" rx=\"12\" fill=\"#ffffff\" stroke=\"#ccdded\"/>\n";
    svg << "<rect x=\"806\" y=\"82\" width=\"440\" height=\"450\" rx=\"12\" fill=\"#ffffff\" stroke=\"#ccdded\"/>\n";

    for (int db = 0; db >= -40; db -= 10) {
        const double r = db_to_r(static_cast<double>(db));
        svg << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << r
            << "\" fill=\"none\" stroke=\"#e4edf7\" stroke-width=\"1\"/>\n";
        svg << "<text x=\"" << (cx + 4.0) << "\" y=\"" << (cy - r - 6.0) << "\" font-family=\"" << svg_font()
            << "\" font-size=\"11\" fill=\"#60758d\">" << db << " dB</text>\n";
    }

    for (int a = -180; a <= 180; a += 30) {
        const auto p1 = pt_xy(static_cast<double>(a), r_inner);
        const auto p2 = pt_xy(static_cast<double>(a), r_outer);
        svg << "<line x1=\"" << p1.first << "\" y1=\"" << p1.second << "\" x2=\"" << p2.first << "\" y2=\"" << p2.second
            << "\" stroke=\"#eef3f9\" stroke-width=\"1\"/>\n";
        const auto pt = pt_xy(static_cast<double>(a), r_outer + 18.0);
        svg << "<text x=\"" << pt.first << "\" y=\"" << (pt.second + 4.0) << "\" text-anchor=\"middle\" font-family=\"" << svg_font()
            << "\" font-size=\"11\" fill=\"#60758d\">" << a << "</text>\n";
    }

    svg << "<polygon points=\"" << poly.str() << "\" fill=\"#0ea5a8\" fill-opacity=\"0.14\"/>\n";
    svg << "<polyline points=\"" << poly.str() << "\" fill=\"none\" stroke=\"url(#strokeA)\" stroke-width=\"2.8\"/>\n";
    svg << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << r_inner << "\" fill=\"#f6f9fd\" stroke=\"#d5e4f2\"/>\n";

    svg << "<text x=\"42\" y=\"40\" font-family=\"" << svg_font() << "\" font-size=\"28\" font-weight=\"700\" fill=\"#10243c\">"
        << html_escape(title) << "</text>\n";
    svg << "<text x=\"42\" y=\"62\" font-family=\"" << svg_font() << "\" font-size=\"13\" fill=\"#4f637b\">Azimute (polar) | Nivel relativo (dB)</text>\n";

    svg << "<text x=\"832\" y=\"122\" font-family=\"" << svg_font() << "\" font-size=\"18\" font-weight=\"700\" fill=\"#0f2034\">Metricas principais</text>\n";
    svg << svg_metrics_block(832.0, 142.0, m);

    svg << "<text x=\"832\" y=\"292\" font-family=\"" << svg_font() << "\" font-size=\"15\" font-weight=\"700\" fill=\"#0f2034\">Legenda</text>\n";
    svg << "<line x1=\"836\" y1=\"316\" x2=\"900\" y2=\"316\" stroke=\"url(#strokeA)\" stroke-width=\"3\"/>\n";
    svg << "<text x=\"912\" y=\"320\" font-family=\"" << svg_font() << "\" font-size=\"12\" fill=\"#35506e\">Perfil do diagrama</text>\n";
    svg << "<rect x=\"836\" y=\"334\" width=\"64\" height=\"12\" fill=\"#0ea5a8\" fill-opacity=\"0.14\" stroke=\"#0ea5a8\"/>\n";
    svg << "<text x=\"912\" y=\"344\" font-family=\"" << svg_font() << "\" font-size=\"12\" fill=\"#35506e\">Area relativa</text>\n";

    svg << "<text x=\"832\" y=\"390\" font-family=\"" << svg_font() << "\" font-size=\"12\" fill=\"#35506e\">Referencia angular:</text>\n";
    svg << "<text x=\"832\" y=\"410\" font-family=\"" << svg_font() << "\" font-size=\"12\" fill=\"#35506e\">0 deg no topo, +90 deg a direita.</text>\n";
    svg << "</svg>\n";

    return svg.str();
}

} // namespace

CutMetrics compute_cut_metrics(const Pattern& cut, PatternKind kind) {
    CutMetrics m;
    if (!is_valid(cut)) {
        m.peak_db = std::nan("");
        m.min_db = std::nan("");
        m.hpbw_deg = std::nan("");
        m.d2d_lin = std::nan("");
        m.d2d_db = std::nan("");
        return m;
    }

    double vmax = 0.0;
    double vmin = 1e30;
    for (std::size_t i = 0; i < cut.values_lin.size(); ++i) {
        const double v = std::max(cut.values_lin[i], 0.0);
        if (v > vmax) {
            vmax = v;
        }
        if (v < vmin) {
            vmin = v;
        }
    }
    m.peak_lin = vmax;
    m.peak_db = (vmax > 0.0) ? 20.0 * std::log10(vmax) : std::nan("");
    m.min_db = (vmin > 0.0) ? 20.0 * std::log10(vmin) : -120.0;
    m.hpbw_deg = hpbw_deg(cut);
    const double span = (kind == PatternKind::Horizontal) ? 360.0 : 180.0;
    m.d2d_lin = directivity_2d_cut(cut, span);
    m.d2d_db = (m.d2d_lin > 0.0) ? 10.0 * std::log10(m.d2d_lin) : std::nan("");
    return m;
}

void write_cut_svg(
    const std::string& path,
    const Pattern& cut,
    PatternKind kind,
    const std::string& title
) {
    if (!is_valid(cut)) {
        throw std::runtime_error("write_cut_svg: invalid cut vectors");
    }
    if (kind == PatternKind::Horizontal) {
        write_text_file(path, render_polar_svg(cut, title));
    } else {
        write_text_file(path, render_planar_svg(cut, title));
    }
}

void write_project_dashboard(
    const std::string& html_path,
    const std::string& project_name,
    const std::vector<DashboardCut>& cuts,
    const std::vector<std::string>& extra_artifacts
) {
    const std::size_t az_count = static_cast<std::size_t>(std::count_if(
        cuts.begin(), cuts.end(), [](const DashboardCut& c) { return c.kind == PatternKind::Horizontal; }
    ));
    const std::size_t el_count = static_cast<std::size_t>(std::count_if(
        cuts.begin(), cuts.end(), [](const DashboardCut& c) { return c.kind == PatternKind::Vertical; }
    ));

    std::vector<std::string> ids(cuts.size());
    for (std::size_t i = 0; i < cuts.size(); ++i) {
        ids[i] = make_id(cuts[i].title, i);
    }

    std::ostringstream html;
    html << "<!doctype html>\n<html lang=\"pt-BR\">\n<head>\n<meta charset=\"utf-8\"/>\n";
    html << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\n";
    html << "<title>" << html_escape(project_name) << " - Dashboard</title>\n";
    html << "<style>\n";
    html << ":root{--bg:#ecf3fa;--surface:#ffffff;--text:#0f172a;--muted:#4b5f76;--line:#d3e1ef;--ink:#10253c;--accent:#0d9488;--accent-deep:#1d4f80;--chip:#e8f2fc;--chip-text:#17466f;}\n";
    html << "*{box-sizing:border-box}html,body{margin:0;padding:0}\n";
    html << "body{font-family:'Bahnschrift','Trebuchet MS','Segoe UI',Arial,sans-serif;background:radial-gradient(circle at 10% 10%,#f8fbff 0,#ecf3fa 55%,#e9f1f9 100%);color:var(--text)}\n";
    html << ".wrap{max-width:1520px;margin:0 auto;padding:22px 22px 30px}\n";
    html << ".hero{position:relative;overflow:hidden;background:linear-gradient(135deg,#11233a,#1b4268);color:#fff;border-radius:18px;padding:26px 28px 24px;box-shadow:0 20px 45px rgba(15,30,52,.22)}\n";
    html << ".hero:after{content:'';position:absolute;right:-80px;top:-60px;width:260px;height:260px;border-radius:50%;background:rgba(13,148,136,.20)}\n";
    html << ".hero h1{margin:0;font-size:31px;letter-spacing:.2px}.hero p{margin:8px 0 0;color:#d6e4f3;font-size:14px}\n";
    html << ".hero-kpi{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px}.hero-kpi .k{padding:7px 11px;border-radius:999px;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25);font-size:12px}\n";
    html << ".layout{display:grid;grid-template-columns:280px 1fr;gap:16px;margin-top:16px}\n";
    html << ".side{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:12px;position:sticky;top:14px;height:fit-content;box-shadow:0 10px 24px rgba(15,23,42,.07)}\n";
    html << ".side h2{margin:0 0 10px;font-size:15px;color:var(--ink)}.side ul{list-style:none;margin:0;padding:0;display:grid;gap:7px}\n";
    html << ".side a{display:block;padding:8px 10px;border-radius:9px;text-decoration:none;color:#16436a;background:#f5f9fe;border:1px solid #dbe8f6;font-size:12px}\n";
    html << ".side a:hover{background:#e8f2fc}\n";
    html << ".main{display:grid;gap:14px}\n";
    html << ".card{background:var(--surface);border:1px solid var(--line);border-radius:14px;box-shadow:0 10px 24px rgba(15,23,42,.06);overflow:hidden}\n";
    html << ".card-h{padding:13px 16px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:8px}\n";
    html << ".card-h h3{margin:0;font-size:17px;color:var(--ink)}.tag{font-size:11px;font-weight:700;letter-spacing:.4px;padding:4px 9px;border-radius:999px;background:var(--chip);color:var(--chip-text);border:1px solid #cfe0f2}\n";
    html << ".card-b{padding:14px 16px}\n";
    html << ".metrics{display:grid;grid-template-columns:repeat(6,minmax(110px,1fr));gap:8px;margin-bottom:12px}\n";
    html << ".m{border:1px solid var(--line);border-radius:10px;padding:9px;background:#fafdff}.m .k{font-size:11px;color:var(--muted)}.m .v{margin-top:4px;font-size:14px;font-weight:700;color:#13263f}\n";
    html << ".plot{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#fff}\n";
    html << ".links{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}.links a{font-size:12px;text-decoration:none;color:#0f3f6d;background:var(--chip);border:1px solid #cfe0f2;padding:5px 9px;border-radius:999px}\n";
    html << ".foot{margin-top:2px;padding:12px 14px;background:var(--surface);border:1px solid var(--line);border-radius:12px}\n";
    html << ".foot h3{margin:0 0 10px 0;font-size:15px}.foot ul{margin:0;padding-left:18px;color:#334155}\n";
    html << "@media(max-width:1260px){.layout{grid-template-columns:1fr}.side{position:static}.metrics{grid-template-columns:repeat(3,minmax(110px,1fr));}}\n";
    html << "@media(max-width:760px){.metrics{grid-template-columns:repeat(2,minmax(110px,1fr));}.hero h1{font-size:26px}}\n";
    html << "</style>\n</head>\n<body>\n<div class=\"wrap\">\n";

    html << "<section class=\"hero\"><h1>" << html_escape(project_name) << "</h1>";
    html << "<p>Painel tecnico consolidado de diagramas, metricas e artefatos gerados automaticamente.</p>";
    html << "<div class=\"hero-kpi\">";
    html << "<span class=\"k\">Cortes: " << cuts.size() << "</span>";
    html << "<span class=\"k\">Azimute: " << az_count << "</span>";
    html << "<span class=\"k\">Elevacao: " << el_count << "</span>";
    html << "<span class=\"k\">Artefatos extras: " << extra_artifacts.size() << "</span>";
    html << "</div></section>\n";

    html << "<section class=\"layout\">\n";
    html << "<aside class=\"side\"><h2>Navegacao rapida</h2><ul>";
    for (std::size_t i = 0; i < cuts.size(); ++i) {
        html << "<li><a href=\"#" << ids[i] << "\">" << html_escape(cuts[i].title) << "</a></li>";
    }
    html << "</ul></aside>\n";
    html << "<div class=\"main\">\n";

    for (std::size_t i = 0; i < cuts.size(); ++i) {
        const DashboardCut& c = cuts[i];
        const CutMetrics m = compute_cut_metrics(c.pattern, c.kind);
        html << "<article id=\"" << ids[i] << "\" class=\"card\">\n";
        html << "<div class=\"card-h\"><h3>" << html_escape(c.title) << "</h3><span class=\"tag\">"
             << ((c.kind == PatternKind::Horizontal) ? "AZIMUTE POLAR" : "ELEVACAO PLANAR") << "</span></div>\n";
        html << "<div class=\"card-b\">\n";
        html << "<div class=\"metrics\">\n";
        html << "<div class=\"m\"><div class=\"k\">Peak (lin)</div><div class=\"v\">" << fmt_num(m.peak_lin, 4) << "</div></div>\n";
        html << "<div class=\"m\"><div class=\"k\">Peak (dB)</div><div class=\"v\">" << fmt_num(m.peak_db, 2) << "</div></div>\n";
        html << "<div class=\"m\"><div class=\"k\">HPBW (deg)</div><div class=\"v\">" << fmt_num(m.hpbw_deg, 2) << "</div></div>\n";
        html << "<div class=\"m\"><div class=\"k\">D2D (lin)</div><div class=\"v\">" << fmt_num(m.d2d_lin, 3) << "</div></div>\n";
        html << "<div class=\"m\"><div class=\"k\">D2D (dB)</div><div class=\"v\">" << fmt_num(m.d2d_db, 2) << "</div></div>\n";
        html << "<div class=\"m\"><div class=\"k\">Min (dB)</div><div class=\"v\">" << fmt_num(m.min_db, 2) << "</div></div>\n";
        html << "</div>\n";
        html << "<img class=\"plot\" src=\"" << html_escape(web_path(c.image_relpath)) << "\" alt=\"" << html_escape(c.title) << "\"/>\n";
        html << "<div class=\"links\">";
        if (!c.table_relpath.empty()) {
            html << "<a href=\"" << html_escape(web_path(c.table_relpath)) << "\">Tabela CSV</a>";
        }
        if (!c.pat_relpath.empty()) {
            html << "<a href=\"" << html_escape(web_path(c.pat_relpath)) << "\">PAT</a>";
        }
        if (!c.adt_relpath.empty()) {
            html << "<a href=\"" << html_escape(web_path(c.adt_relpath)) << "\">PAT ADT</a>";
        }
        html << "</div>\n";
        html << "</div>\n";
        html << "</article>\n";
    }

    if (!extra_artifacts.empty()) {
        html << "<section class=\"foot\"><h3>Artefatos adicionais</h3><ul>";
        for (std::size_t i = 0; i < extra_artifacts.size(); ++i) {
            html << "<li><a href=\"" << html_escape(web_path(extra_artifacts[i])) << "\">"
                 << html_escape(extra_artifacts[i]) << "</a></li>";
        }
        html << "</ul></section>\n";
    }

    html << "</div></section>\n";
    html << "</div>\n</body>\n</html>\n";
    write_text_file(html_path, html.str());
}

} // namespace eftx

