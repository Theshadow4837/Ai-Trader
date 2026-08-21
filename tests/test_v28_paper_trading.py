import sys
import tempfile
import unittest
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v28_paper_account import PaperPortfolio, TRANSACTION_COST_RATE, load_account
import v28_paper_runner as runner
import multi_data_loader
import v28_validation
import v28_paper_report


class PaperPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.paths = (base / "account.json", base / "equity.csv", base / "trades.csv")
        self.portfolio = PaperPortfolio(*self.paths)

    def tearDown(self):
        self.temp.cleanup()

    def test_position_change_creates_one_trade_and_one_cost(self):
        self.portfolio.process("2026-01-02", 1, 100.0)
        trades = pd.read_csv(self.paths[2])
        account = load_account(self.paths[0])
        self.assertEqual(len(trades), 1)
        self.assertEqual(account.trade_count, 1)
        self.assertAlmostEqual(account.transaction_costs, 10_000 * TRANSACTION_COST_RATE / (1 + TRANSACTION_COST_RATE))

    def test_duplicate_date_does_not_advance_or_charge_twice(self):
        self.portfolio.process("2026-01-02", 1, 100.0)
        before = self.paths[0].read_text()
        with self.assertRaisesRegex(RuntimeError, "already processed"):
            self.portfolio.process("2026-01-02", 1, 101.0)
        self.assertEqual(before, self.paths[0].read_text())
        self.assertEqual(len(pd.read_csv(self.paths[1])), 1)

    def test_backwards_date_raises(self):
        self.portfolio.process("2026-01-03", 0, 100.0)
        with self.assertRaisesRegex(RuntimeError, "moved backwards"):
            self.portfolio.process("2026-01-02", 0, 100.0)

    def test_state_survives_restart_and_exit_is_one_trade(self):
        self.portfolio.process("2026-01-02", 1, 100.0)
        restarted = PaperPortfolio(*self.paths)
        restarted.process("2026-01-03", 0, 110.0)
        account = load_account(self.paths[0])
        self.assertEqual(account.position, 0)
        self.assertEqual(account.trade_count, 2)
        self.assertEqual(len(pd.read_csv(self.paths[2])), 2)
        self.assertGreater(account.realized_pnl, 0)

    def test_all_position_transitions_are_deterministic(self):
        flat = self.portfolio.process("2026-01-02", 0, 100.0)
        self.assertEqual(flat["position"], 0)
        self.assertFalse(self.paths[2].exists())

        enter = self.portfolio.process("2026-01-03", 1, 100.0)
        self.assertEqual(enter["position"], 1)
        self.assertEqual(len(pd.read_csv(self.paths[2])), 1)

        hold = self.portfolio.process("2026-01-04", 1, 105.0)
        self.assertEqual(hold["position"], 1)
        self.assertEqual(len(pd.read_csv(self.paths[2])), 1)
        self.assertEqual(hold["transaction_cost"], 0.0)

        exit_row = self.portfolio.process("2026-01-05", 0, 110.0)
        self.assertEqual(exit_row["position"], 0)
        self.assertEqual(len(pd.read_csv(self.paths[2])), 2)

        stay_flat = self.portfolio.process("2026-01-06", 0, 111.0)
        self.assertEqual(stay_flat["position"], 0)
        self.assertEqual(len(pd.read_csv(self.paths[2])), 2)


class RunnerValidationTests(unittest.TestCase):
    def test_frozen_hash_and_feature_schema(self):
        self.assertEqual(v28_validation.sha256_file(ROOT / "models/v28/v28_seed_202_FROZEN.zip"), runner.EXPECTED_MODEL_SHA256)
        df = runner.load_data()
        features = runner.find_features(df)
        self.assertEqual(len(features), 85)
        runner.validate_feature_schema(df, features)
        comparison = v28_validation.compare_feature_schema(df, runner.REFERENCE_FEATURE_FILE)
        self.assertFalse(comparison["missing"])
        self.assertFalse(comparison["extra"])
        self.assertTrue(comparison["order_match"])

    def test_missing_extra_and_future_columns_are_rejected(self):
        df = runner.load_data().tail(1).copy()
        with self.assertRaisesRegex(RuntimeError, "exactly 85"):
            runner.validate_feature_schema(df.drop(columns=["SPY_return_1d"]), runner.find_features(df.drop(columns=["SPY_return_1d"])))
        extra = df.assign(unexpected_feature=1.0)
        with self.assertRaisesRegex(RuntimeError, "exactly 85"):
            runner.validate_feature_schema(extra, runner.find_features(extra))
        future = df.assign(future_leak=1.0)
        with self.assertRaisesRegex(RuntimeError, "Forbidden"):
            runner.validate_feature_schema(future, runner.find_features(future))

    def test_nan_new_observation_and_regressing_data_are_rejected(self):
        df = runner.load_data().tail(1).copy()
        df.loc[df.index[0], "SPY_return_1d"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            runner.get_new_rows(df, runner.find_features(df), pd.DataFrame())
        data = pd.DataFrame({"Date": pd.to_datetime(["2026-01-02"])})
        log = pd.DataFrame({"date": pd.to_datetime(["2026-01-03"])})
        with self.assertRaisesRegex(RuntimeError, "STALE MARKET DATA"):
            runner.validate_market_progress(data, log)

    def test_stale_data_yields_no_new_rows(self):
        df = runner.load_data().tail(1).copy()
        existing = pd.DataFrame({"date": pd.to_datetime(df["Date"])})
        self.assertTrue(runner.get_new_rows(df, runner.find_features(df), existing).empty)

    def test_training_normalization_is_finite_and_clipped(self):
        df = runner.load_data()
        features = runner.find_features(df)
        mean, std = runner.get_training_stats(df, features)
        self.assertEqual(mean.shape, (85,))
        self.assertEqual(std.shape, (85,))
        self.assertTrue((std > 0).all())
        X = runner.normalize_features(df.tail(3), features, mean, std)
        self.assertEqual(X.shape, (3, 85))
        self.assertTrue((X <= 10.0).all())
        self.assertTrue((X >= -10.0).all())

    def test_future_return_columns_cannot_enter_inference_features(self):
        df = runner.load_data().tail(1).copy()
        df["future_1d_return"] = 0.99
        df["target"] = 1
        features = runner.find_features(df)
        self.assertNotIn("future_1d_return", features)
        self.assertNotIn("target", features)
        with self.assertRaisesRegex(RuntimeError, "Forbidden"):
            runner.validate_feature_schema(df, features)

    def test_running_twice_leaves_account_state_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            portfolio = PaperPortfolio(base / "account.json", base / "equity.csv", base / "trades.csv")
            portfolio.process("2026-01-02", 1, 100.0)
            before = (base / "account.json").read_text()
            with self.assertRaisesRegex(RuntimeError, "already processed"):
                portfolio.process("2026-01-02", 1, 100.0)
            self.assertEqual(before, (base / "account.json").read_text())

    def test_equity_and_drawdown_calculation(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            portfolio = PaperPortfolio(base / "account.json", base / "equity.csv", base / "trades.csv")
            portfolio.process("2026-01-02", 1, 100.0)
            portfolio.process("2026-01-03", 1, 90.0)
            account = load_account(base / "account.json")
            self.assertLess(account.current_equity, account.starting_capital)
            self.assertLess(account.maximum_drawdown, 0.0)

    def test_market_data_cleaner_rejects_malformed_rows_and_removes_duplicates(self):
        raw = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [10, 11, 12],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-03"]),
        )
        cleaned = multi_data_loader.clean_market_data(raw, "SPY")
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned["Date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-01-02", "2026-01-03"])
        broken = raw.copy()
        broken.loc[broken.index[0], "Close"] = None
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            multi_data_loader.clean_market_data(broken, "SPY")

    def test_paper_execution_path_has_no_network_or_broker_api(self):
        for name in ["v28_paper_account.py", "v28_paper_runner.py"]:
            source = (ROOT / "src" / name).read_text().lower()
            self.assertIsNone(re.search(r"^(?:from|import)\s+(?:requests|alpaca|ibkr|ccxt)\b", source, re.MULTILINE))
            self.assertNotRegex(source, r"\b(?:submit_order|place_order)\s*\(")

    def test_frozen_model_file_is_not_modified_by_validation(self):
        model = ROOT / "models/v28/v28_seed_202_FROZEN.zip"
        before = model.stat().st_mtime_ns
        v28_validation.verify_model_hash(model)
        after = model.stat().st_mtime_ns
        self.assertEqual(before, after)

    def test_report_metrics_and_trade_returns(self):
        equity = pd.DataFrame({"current_equity": [100.0, 110.0, 105.0, 120.0]})
        trades = pd.DataFrame(
            [
                {"date": "2026-01-02", "side": "BUY", "notional": 100.0, "transaction_cost": 1.0, "realized_pnl": 0.0},
                {"date": "2026-01-03", "side": "SELL", "notional": 111.0, "transaction_cost": 1.0, "realized_pnl": 9.0},
                {"date": "2026-01-04", "side": "BUY", "notional": 110.0, "transaction_cost": 1.0, "realized_pnl": 0.0},
                {"date": "2026-01-05", "side": "SELL", "notional": 104.0, "transaction_cost": 1.0, "realized_pnl": -7.0},
            ]
        )
        returns = v28_paper_report.closed_trade_returns(trades)
        self.assertEqual(len(returns), 2)
        self.assertGreater(v28_paper_report.annualized_return(equity), 0)
        self.assertNotEqual(v28_paper_report.sharpe_ratio(equity), 0)
        self.assertEqual(v28_paper_report.longest_drawdown_days(equity), 1)
        self.assertGreater(v28_paper_report.turnover(equity, trades), 0)


if __name__ == "__main__":
    unittest.main()
