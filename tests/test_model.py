import numpy as np
import torch

from nifty_mtl.config import Config, ModelConfig
from nifty_mtl.models.dataset import Scalers, WeekDataset, target_returns
from nifty_mtl.models.losses import mtl_loss, soft_weights
from nifty_mtl.models.mtl import MTLModel, denormalise


def test_model_output_shape():
    model = MTLModel(f_seq=31, f_static=41, lookback=60, cfg=ModelConfig())
    seq, static = torch.randn(32, 60, 31), torch.randn(32, 41)
    r, v = model(seq, static)
    assert r.shape == (32,) and v.shape == (32,)


def test_vol_clip():
    class S:  # minimal scalers stub
        ret_mean, ret_std, vol_mean, vol_std = 0.0, 0.03, 0.03, 0.02
    r, v = denormalise(torch.zeros(5), torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0]), S())
    assert v.min() >= 0.01 and v.max() <= 0.10


def test_attention_weights_are_distributions():
    model = MTLModel(31, 41, 60, ModelConfig()).eval()
    model(torch.randn(4, 60, 31), torch.randn(4, 41), need_weights=True)
    maps = model.attention_maps()
    assert len(maps) == 2 and maps[0].shape == (4, 8, 60, 60)
    assert torch.allclose(maps[0].sum(-1), torch.ones(4, 8, 60), atol=1e-5)
    assert torch.allclose(model.last_pool_weights.sum(-1), torch.ones(4), atol=1e-5)


def test_loss_cost_term_penalises_turnover():
    class S:
        ret_mean, ret_std, vol_mean, vol_std = 0.0, 0.03, 0.03, 0.02
    y_ret, y_vol = torch.zeros(10), torch.full((10,), 0.03)
    r = torch.linspace(-1, 1, 10); v = torch.zeros(10)
    # identical previous-week predictions -> zero turnover
    l_same, p_same = mtl_loss(r, v, y_ret, y_vol, S(), lam=1.0, prev=(r, v))
    # reversed previous-week predictions -> large turnover
    l_flip, p_flip = mtl_loss(r, v, y_ret, y_vol, S(), lam=1.0, prev=(-r, v))
    assert p_same["turnover"] < 1e-5 and p_flip["turnover"] > 0.5
    assert l_flip > l_same


def test_soft_weights_sum_to_one():
    w = soft_weights(torch.randn(20), tau=0.1)
    assert abs(float(w.sum()) - 1) < 1e-5 and (w >= 0).all()


def test_scalers_fit_only_on_train(features):
    T = features.seq.shape[0]
    first = int(np.argmax(features.sample_ok.any(1)))
    train = np.zeros(T, bool); train[first: first + 40] = True
    sc = Scalers.fit(features, train)
    assert np.all(sc.static_std[features.n_graph:] == 1.0)  # one-hot untouched
    ds = WeekDataset(features, sc, np.where(train)[0])
    b = ds[0]
    assert torch.isfinite(b["seq"]).all() and torch.isfinite(b["static"]).all()


def test_cs_demean_target_has_zero_weekly_mean(features):
    y = target_returns(features, "cs_demean")
    ok = features.sample_ok
    for t in np.where(ok.sum(1) > 5)[0][:20]:
        assert abs(y[t, ok[t]].mean()) < 1e-5
