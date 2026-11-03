"""P1 exit criteria on a real trained network (examples/mlp_digits.py)."""
import pytest

import mlp_digits


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_p1_exit_criteria(seed):
    rep, _ = mlp_digits.evaluate(seed)
    assert rep.hidden_int8_mismatch <= 1e-3, rep
    assert rep.logit_err_vs_fakequant <= 1e-3, rep
    assert rep.top1_golden_vs_fakequant >= 0.99, rep
    assert rep.acc_float - rep.acc_golden <= 0.01, rep
    assert rep.exit_ok()


@pytest.mark.parametrize("variant", [dict(fused=False), dict(m_split=64), dict(base_addr=1 << 21),
                                     dict(n_split=32), dict(n_split=16, m_split=100, fused=False)])
def test_mlp_logits_do_not_depend_on_program_shape(variant):
    m = mlp_digits.train(0)
    q = mlp_digits.calibrate(m)
    _, _, _, h0, l0 = mlp_digits.compile_and_run(m, q, m.x_test)
    _, _, _, h1, l1 = mlp_digits.compile_and_run(m, q, m.x_test, **variant)
    assert h0.tobytes() == h1.tobytes() and l0.tobytes() == l1.tobytes()
