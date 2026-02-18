from core.math_engine import MarkerValue, evaluate_expression


def test_math_engine_valid_expression():
    A = MarkerValue(name="A", ang_deg=10.0, mag_lin=1.0, mag_db=0.0)
    B = MarkerValue(name="B", ang_deg=-170.0, mag_lin=0.5, mag_db=-6.0206)
    v = evaluate_expression("ang_dist(A.ang_deg, B.ang_deg)", context={"A": A, "B": B, "params": {"xdb": 10}})
    assert abs(float(v) - 180.0) < 1e-9


def test_math_engine_blocks_unsafe_nodes():
    failed = False
    try:
        evaluate_expression("__import__('os').system('echo hi')", context={})
    except Exception:
        failed = True
    assert failed


def test_math_engine_params_subscript():
    A = MarkerValue(name="A", ang_deg=0.0, mag_lin=1.0, mag_db=0.0)
    B = MarkerValue(name="B", ang_deg=10.0, mag_lin=0.7, mag_db=-3.0)
    v = evaluate_expression("params['xdb'] + (B.mag_db - A.mag_db)", context={"A": A, "B": B, "params": {"xdb": 10}})
    assert abs(float(v) - 7.0) < 1e-9
