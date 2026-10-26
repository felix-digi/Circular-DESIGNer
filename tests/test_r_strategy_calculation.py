"""Regression checks for zero R7 scores and malformed positional weights."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QWidget

from tool import CalculationDialog, Config, GoalDialog, MainWindow, ResultWindow


ROOT = Path(__file__).resolve().parents[1]
CORRECTED_WEIGHTS = ROOT / "tests/fixtures/goal_to_r_strategy.corrected.json"


def malformed_weights():
    return {"Minimise disassembly time": [
        0.0, 0.0, 0.0, 0.0, 2.218838544, 1.282473043,
        1, 172317705, 0.0, 0.339962357, 0.0,
    ]}


class RStrategyCalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        config = Config()
        for name in ("materials_json", "parameter_to_goal_json", "goal_to_r_strategy_json",
                     "joining_connections_json", "classification_json"):
            setattr(config, name, str(ROOT / getattr(config, name)))
        self.calc = CalculationDialog({}, "", config)
        self.addCleanup(self.calc.close)
        self.weights = json.loads(CORRECTED_WEIGHTS.read_text())

    def test_corrected_uploaded_weights_keep_zero_upcycling_at_zero_percent(self):
        self.assertEqual(
            [(goal, row[7]) for goal, row in self.weights.items() if row[7]],
            [("Potential for upcycling", 4.0)],
        )
        self.calc.goal_to_r_strategy = self.weights
        self.calc.parameter_upcycling_potential = 0.0
        self.calc.parameter_disassembly_time = 1.0
        self.calc.calculate_goal_achievement()
        self.assertEqual(self.calc.goal_scores["Potential for upcycling"], 0.0)
        self.assertEqual(self.calc.goal_scores["Minimise disassembly time"], 1.0)
        scores = self.calc.calculate_r_strategy_scores()
        self.assertEqual(scores["R7"], 0.0)

        figure = Figure()
        chart = SimpleNamespace(r_strategy_scores=scores, canvas_r_strategy=figure.canvas)
        ResultWindow.plot_r_strategy_scores(chart, figure)
        self.assertEqual(figure.axes[0].patches[7].get_height(), 0.0)
        self.assertEqual(figure.axes[0].texts[7].get_text(), "0.00")

    def test_corrected_weights_preserve_fractional_upcycling_score(self):
        self.calc.goal_to_r_strategy = self.weights
        self.calc.goal_scores = {goal: 1.0 for goal in self.weights}
        self.calc.goal_scores["Potential for upcycling"] = 0.25
        self.assertEqual(self.calc.calculate_r_strategy_scores()["R7"], 0.25)

    def test_direct_scorer_rejects_malformed_rows_and_clears_old_scores(self):
        self.calc.goal_to_r_strategy = malformed_weights()
        self.calc.goal_scores = {}  # Even uncalculated goals must be validated.
        self.calc.r_strategy_scores = {"R7": 1.0}
        with self.assertRaisesRegex(ValueError, "Minimise disassembly time.*found 11"):
            self.calc.calculate_r_strategy_scores()
        self.assertEqual(self.calc.r_strategy_scores, {})

    def test_start_calculation_stops_once_before_any_other_calculation(self):
        self.calc.goal_to_r_strategy = malformed_weights()
        with patch("tool.QMessageBox.warning") as warning, \
             patch("tool.QMessageBox.critical") as critical, \
             patch.object(self.calc, "calculate_classification_averages") as next_step:
            self.calc.start_calculation()
        warning.assert_called_once()
        critical.assert_not_called()
        next_step.assert_not_called()
        self.assertIn("decimal point", warning.call_args.args[2])
        self.assertIn("stopped", self.calc.status_bar.text().lower())
        self.assertEqual(self.calc.r_strategy_scores, {})

    def test_weight_editor_rejects_malformed_file_before_populating_controls(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            path.write_text(json.dumps(malformed_weights()))
            config = SimpleNamespace(goal_to_r_strategy_json=str(path))
            with self.assertRaisesRegex(ValueError, "found 11"):
                GoalDialog(config)

        parent = QWidget()
        self.addCleanup(parent.close)
        with patch("tool.GoalDialog", side_effect=ValueError("Invalid goal weights")), \
             patch("tool.QMessageBox.warning") as warning:
            MainWindow.open_goal_assessment(parent)
        warning.assert_called_once()

    def test_weight_editor_does_not_save_invalid_values(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            path.write_text(json.dumps(self.weights))
            before = path.read_bytes()
            dialog = GoalDialog(SimpleNamespace(goal_to_r_strategy_json=str(path)))
            self.addCleanup(dialog.close)
            dialog.goal_data["Potential for upcycling"][7] = 172317705
            with patch("tool.QMessageBox.warning") as warning:
                dialog.save_changes()
            warning.assert_called_once()
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
