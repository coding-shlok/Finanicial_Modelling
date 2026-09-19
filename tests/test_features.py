import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from nifty_mtl.data.preprocess import build_weekly
from nifty_mtl.features.build import SEQ_NAMES, build_features
from nifty_mtl.features.graph import GRAPH_NAMES
from nifty_mtl.features.technical import rsi

from .conftest import make_daily


def test_weekly_panel_shapes(panel):
    assert panel.close.shape[1] == 12
    assert panel.close.index.freqstr in ("W-FRI", None)
    assert all(d.dayofweek == 4 for d in panel.weeks)  # Fridays


def test_weekly_return_definition(panel):
    c = panel.close.iloc[:, 0]
    expected = np.log(c / c.shift(1))
    pd.testing.assert_series_equal(panel.ret.iloc[1:, 0], expected.iloc[1:], check_names=False)


def test_targets_are_next_week(panel):
    y_ret, y_vol = panel.targets()
    # target at t equals realized return of week t+1
    assert np.allclose(y_ret.iloc[10].values, panel.ret.iloc[11].values, equal_nan=True)
    assert y_vol.max().max() <= 0.10 + 1e-9 and y_vol.min().min() >= 0.01 - 1e-9


def test_feature_no_nan(features):
    ok = features.sample_ok
    assert ok.sum() > 0
    assert np.isfinite(features.seq[ok]).all()
    assert np.isfinite(features.static[ok]).all()
    assert np.isfinite(features.y_ret[ok]).all()
    assert np.isfinite(features.y_vol[ok]).all()


def test_feature_counts(features):
    assert features.seq.shape[-1] == len(SEQ_NAMES) == 31
    assert features.static.shape[-1] == len(GRAPH_NAMES) + 4  # 30 graph + 4 sector one-hot
    assert (features.static[..., -4:].sum(-1) == 1).all()


def test_feature_normalization(features):
    ok = features.sample_ok
    X = features.static[ok][:, :30]
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    assert abs(Xn.mean()) < 1e-6
    assert abs(Xn.std() - 1) < 1e-2


def test_rsi_bounds():
    rng = np.random.default_rng(1)
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (200, 3)), axis=0)))
    r = rsi(c).dropna()
    assert ((r >= 0) & (r <= 100)).all().all()


def test_no_look_ahead_bias(universe):
    """Perturbing all prices after week t must leave features/targets at <= t-1 unchanged.

    Features at t (used to predict t+1) may depend on t's bar, so we check
    strictly earlier weeks and the target at t-1 (= return of week t) is
    unaffected as well because the perturbation starts *after* week t.
    """
    base = make_daily(seed=3)
    panel_a = build_weekly(base, universe)
    fa = build_features(panel_a, universe)

    t = fa.weeks[len(fa.weeks) // 2]
    cutoff = t + pd.Timedelta(days=1)
    pert = base.copy()
    later = pert.index.get_level_values("date") > cutoff
    for col in ["Open", "High", "Low", "Close"]:
        pert.loc[later, col] *= 1.37
    pert.loc[later, "Volume"] *= 3.0
    panel_b = build_weekly(pert, universe)
    fb = build_features(panel_b, universe)

    ti = fa.week_index(str(t.date()))
    assert (fa.weeks == fb.weeks).all()
    np.testing.assert_array_equal(fa.seq[: ti + 1], fb.seq[: ti + 1])
    np.testing.assert_array_equal(fa.static[: ti + 1], fb.static[: ti + 1])
    # target at t-1 is the return of week t which is untouched
    np.testing.assert_array_equal(fa.y_ret[:ti], fb.y_ret[:ti])
    np.testing.assert_array_equal(fa.y_vol[:ti], fb.y_vol[:ti])
    # and the perturbation *is* visible after t (sanity that the test has teeth)
    assert not np.array_equal(fa.y_ret[ti], fb.y_ret[ti])
