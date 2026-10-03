import unittest

import numpy as np
import pandas as pd
import torch

from run import FEATURES, ReturnNetwork, normalize_monthly, portfolio_series


class ExperimentTests(unittest.TestCase):
    def test_normalization_does_not_read_returns_or_future_months(self):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(rng.normal(size=(40, len(FEATURES))), columns=FEATURES)
        frame['month'] = ['2020-01']*20+['2020-02']*20
        frame['actual'] = rng.normal(size=40)
        frame.loc[0, 'ep'] = np.nan
        first = normalize_monthly(frame)
        frame['actual'] = 1000
        frame.loc[20:, FEATURES] = 999
        second = normalize_monthly(frame)
        np.testing.assert_array_equal(first[:20], second[:20])
        self.assertEqual(first.shape, (40,32))
        self.assertTrue(np.isfinite(first).all())
        self.assertEqual(first[0, len(FEATURES)+FEATURES.index('ep')], 1)

    def test_decile_returns_and_initial_trade_cost(self):
        frame = pd.DataFrame({'asset_id':[f'{i:06}' for i in range(20)],
                              'outcome_month':'2023-01', 'prediction':np.arange(20),
                              'actual':np.arange(20)/100, 'cap':1})
        result = portfolio_series(frame,'prediction')
        np.testing.assert_allclose(result.spread, .18)
        np.testing.assert_allclose(result.target_weight_turnover, 2)
        np.testing.assert_allclose(result.net_10bps, .178)

    def test_encoder_receives_return_prediction_gradient(self):
        torch.manual_seed(42)
        network = ReturnNetwork(32,True)
        x = torch.randn(50,32)
        loss = (network(x)-torch.randn(50)).square().mean()
        loss.backward()
        self.assertEqual(network.encoder(x).shape, (50,32))
        self.assertGreater(network.encoder[0].weight.grad.abs().sum().item(),0)


if __name__ == '__main__':
    unittest.main()
