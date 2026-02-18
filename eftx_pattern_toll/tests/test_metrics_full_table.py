from reports.metrics import metrics_items_from_dict


def test_metrics_items_keep_known_and_extra_fields():
    metrics = {
        "kind": "H",
        "peak_db": 0.0,
        "peak_angle_deg": 10.0,
        "hpbw_deg": 35.5,
        "d2d": 7.1234,
        "d2d_db": 8.53,
        "first_null_db": -19.8,
        "fb_db": 24.1,
        "angle_min": -180.0,
        "angle_max": 180.0,
        "step_deg": 1.0,
        "points": 361,
        "custom_metric_alpha": 12.345,
        "solver_name": "HFSS",
    }

    items = metrics_items_from_dict(metrics)
    labels = [k for k, _ in items]

    assert "Peak" in labels
    assert "HPBW" in labels
    assert "Points" in labels
    assert "Custom Metric Alpha" in labels
    assert "Solver Name" in labels
