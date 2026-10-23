"""Run with: QT_QPA_PLATFORM=offscreen python -B -m unittest discover -s tests."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QLabel

from tool import (
    CalculationDialog,
    GoalDialog,
    ParameterAssignmentDialog,
    ParameterResultsDialog,
    ResultWindow,
)


ROOT = Path(__file__).resolve().parents[1]
FIELD = "incineration_for_energy_recovery"
PARAMETER = "parameter_incineration_for_energy_recovery"
GOAL = "Potential for energy recovery by incineration"


def component(material, mass, **extra):
    return {"material": material, "mass [kg]": mass, **extra}


class EnergyRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        materials = {
            "yes": {FIELD: True},
            "no": {FIELD: False},
            "unknown": {FIELD: None},
            "legacy": {},
            "string_flag": {FIELD: "true"},
            "numeric_flag": {FIELD: 1},
        }
        (path / "materials.json").write_text(json.dumps({"materials": materials}))
        (path / "empty.json").write_text("{}")
        for name in ("parameter_to_goal.json", "goal_to_r_strategy.json"):
            (path / name).write_bytes((ROOT / "app_data" / name).read_bytes())
        self.config = SimpleNamespace(
            materials_json=str(path / "materials.json"),
            parameter_to_goal_json=str(path / "parameter_to_goal.json"),
            goal_to_r_strategy_json=str(path / "goal_to_r_strategy.json"),
            joining_connections_json=str(path / "empty.json"),
            classification_json=str(path / "empty.json"),
        )
        self.calc = self.widget(CalculationDialog({}, "", self.config))

    def widget(self, widget):
        self.addCleanup(widget.close)
        return widget

    def calculate(self, roots):
        self.calc.data = {"data": roots}
        value = self.calc.calculate_parameter_incineration_for_energy_recovery()
        self.assertEqual(self.calc.get_all_parameters()[PARAMETER], value)
        return value

    def test_share_uses_mass_and_accepts_component_roots(self):
        self.assertAlmostEqual(
            self.calculate([component("yes", 2), component("no", 8)]), 0.2
        )

    def test_nested_assembly_totals_are_not_counted_again(self):
        roots = [component("yes", 1000, components=[
            component("no", 700, components=[
                component("yes", 2), component("no", 6),
            ]),
            component("yes", 2),
        ])]
        self.assertAlmostEqual(self.calculate(roots), 0.4)

    def test_unknown_and_unassigned_materials_remain_in_denominator(self):
        roots = [component(name, 2) for name in (
            "yes", "no", "unknown", "legacy", "not_in_database",
            "string_flag", "numeric_flag", None,
        )]
        roots.append({"mass [kg]": 2})
        self.assertAlmostEqual(self.calculate(roots), 2 / 18)

    def test_total_mass_fallback_only_for_missing_or_none_mass(self):
        roots = [
            {"material": "yes", "total mass [kg]": 3},
            component("no", None, **{"total mass [kg]": 7}),
            component("yes", 0, **{"total mass [kg]": 100}),
            component("yes", -2, **{"total mass [kg]": 100}),
        ]
        self.assertAlmostEqual(self.calculate(roots), 0.3)

    def test_invalid_masses_are_ignored(self):
        for mass in (0, -2, float("inf"), float("-inf"), float("nan"),
                     True, False, "2", [], {}):
            with self.subTest(mass=mass):
                roots = [component("yes", 2), component("no", 8),
                         component("no", mass)]
                self.assertAlmostEqual(self.calculate(roots), 0.2)

    def test_no_positive_mass_is_unavailable_and_no_eligible_mass_is_zero(self):
        for roots in ([], [component("yes", None)], [component("yes", 0)]):
            with self.subTest(roots=roots):
                self.assertIsNone(self.calculate(roots))
        self.assertEqual(self.calculate([component("not_in_database", 3)]), 0.0)

    def test_goal_mapping_uses_the_calculated_mass_share(self):
        self.assertEqual(self.calc.parameter_to_goal["End of Life"][GOAL], [PARAMETER])
        self.calculate([component("yes", 3), component("no", 7)])
        self.assertAlmostEqual(self.calc.calculate_goal_achievement()[GOAL], 0.3)
        self.calculate([])
        self.assertIsNone(self.calc.calculate_goal_achievement()[GOAL])

    def test_default_zero_weights_preserve_all_strategy_scores(self):
        weights = self.calc.goal_to_r_strategy.pop(GOAL)
        self.assertEqual(weights, [0.0] * 10)
        self.assertTrue(all(type(weight) is float for weight in weights))
        self.calc.goal_scores = {goal: 0.6 for goal in self.calc.goal_to_r_strategy}
        before = self.calc.calculate_r_strategy_scores().copy()
        self.calc.goal_to_r_strategy[GOAL] = weights
        self.calc.goal_scores[GOAL] = 0.3
        self.assertEqual(self.calc.calculate_r_strategy_scores(), before)

    def test_nonzero_weights_affect_only_selected_strategies(self):
        original = copy.deepcopy(self.calc.goal_to_r_strategy)
        self.calc.goal_scores = {goal: 0.6 for goal in original}
        self.calc.goal_scores[GOAL] = 0.3
        baseline = self.calc.calculate_r_strategy_scores().copy()
        for selected in ({9: 2.0}, {2: 1.5, 9: 3.0}):
            with self.subTest(selected=selected):
                self.calc.goal_to_r_strategy = copy.deepcopy(original)
                for index, weight in selected.items():
                    self.calc.goal_to_r_strategy[GOAL][index] = weight
                scores = self.calc.calculate_r_strategy_scores()
                for index in range(10):
                    if index not in selected:
                        self.assertEqual(scores[f"R{index}"], baseline[f"R{index}"])
                        continue
                    previous_weight = sum(abs(values[index]) for goal, values in original.items()
                                          if goal != GOAL)
                    previous_score = sum(0.6 * values[index] for goal, values in original.items()
                                         if goal != GOAL)
                    weight = selected[index]
                    expected = (previous_score + 0.3 * weight) / (previous_weight + abs(weight))
                    self.assertAlmostEqual(scores[f"R{index}"], expected)

    def test_result_chart_displays_weighted_scores_as_percentages(self):
        self.calculate([component("yes", 3), component("no", 7)])
        self.calc.calculate_goal_achievement()
        self.calc.goal_to_r_strategy = {GOAL: [0.0] * 10}
        self.calc.goal_to_r_strategy[GOAL][2] = 1.5
        self.calc.goal_to_r_strategy[GOAL][9] = 3.0
        scores = self.calc.calculate_r_strategy_scores()
        figure = Figure()
        chart = SimpleNamespace(r_strategy_scores=scores, canvas_r_strategy=figure.canvas)
        ResultWindow.plot_r_strategy_scores(chart, figure)
        heights = [bar.get_height() for bar in figure.axes[0].patches]
        self.assertEqual(len(heights), 10)
        for index, height in enumerate(heights):
            self.assertAlmostEqual(height, 30.0 if index in (2, 9) else 0.0)

    def test_parameter_results_show_unavailable_and_numeric_values(self):
        for value, text in ((None, "N/A"), (0.3, "0.3")):
            with self.subTest(value=value):
                dialog = self.widget(ParameterResultsDialog({PARAMETER: value}))
                labels = [label.text() for label in dialog.findChildren(QLabel)]
                self.assertTrue(any("incineration for energy recovery" in label
                                    and label.endswith(text) for label in labels))

    def test_editors_expose_parameter_and_save_independent_weights(self):
        assignments = self.widget(ParameterAssignmentDialog(self.config))
        tree = assignments.goal_tree
        end_of_life = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
                           if tree.topLevelItem(i).text(0) == "End of Life")
        goal_item = next(end_of_life.child(i) for i in range(end_of_life.childCount())
                         if end_of_life.child(i).text(0) == GOAL)
        tree.setCurrentItem(goal_item)
        self.assertTrue(assignments.checkboxes[PARAMETER].isChecked())

        weights = self.widget(GoalDialog(self.config))
        row = next(row for row in range(weights.table.rowCount())
                   if weights.table.item(row, 0).text() == GOAL)
        for index in range(10):
            self.assertEqual(weights.table.cellWidget(row, index + 1).value(), 0.0)
        weights.table.cellWidget(row, 3).setValue(1.5)  # R2
        weights.table.cellWidget(row, 10).setValue(3.0)  # R9
        with patch("tool.QMessageBox.information"):
            weights.save_changes()
        saved = json.loads(Path(self.config.goal_to_r_strategy_json).read_text())
        self.assertEqual(saved[GOAL], [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0])


if __name__ == "__main__":
    unittest.main()
