"""Regression checks for derived assembly classifications in the calculation UI."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PyQt5.QtWidgets import QApplication

from tool import CalculationDialog, SelectionDialog


ROOT = Path(__file__).resolve().parents[1]


def reference(components=28):
    return {
        "used_materials": ["DC_01"],
        "number_of_components": components,
        "number_of_materials": 1,
        "number_joining_connections": 93,
        "number_of_different_joining_connections": 50,
        "number_of_tools": 11,
        "disassembly_time [s]": "26520.5",
        "assembly_time [s]": "7951.63",
    }


def part(classification):
    return {
        "name": classification, "id": classification,
        "components": [], "material": "DC_01",
        "mass [kg]": 1.0, "total mass [kg]": 1.0,
        "volume [mm^3]": 1.0, "density [kg/mm^3]": 1.0,
        "basic_parameters": {
            "classification": classification,
            "lifespan [a]": 10,
            "standard_component": "Yes",
        },
        "disassembly_parameters": {"assigned_connections": {}},
    }


class ClassificationCalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        self.hierarchy = {"Vehicle": {"Body": {"Tailgate": {
            "Examples": [reference()],
            "Spoiler": {"Examples": [reference(1)]},
            "Hinge assembly": {
                "Examples": [reference(2)],
                "Hinge pin": {"Examples": [reference(1)]},
                "Hinge arm": {"Examples": [reference(1)]},
            },
        }}}}
        self.classification_path = path / "classification.json"
        self.classification_path.write_text(json.dumps(self.hierarchy), encoding="utf-8")
        self.config = SimpleNamespace(
            classification_json=str(self.classification_path),
            materials_json=str(ROOT / "app_data/materials_validation.json"),
            parameter_to_goal_json=str(ROOT / "app_data/parameter_to_goal.json"),
            goal_to_r_strategy_json=str(ROOT / "app_data/goal_to_r_strategy.json"),
            joining_connections_json=str(ROOT / "joining_connections.json"),
        )
        spoiler, pin, arm = part("Spoiler"), part("Hinge pin"), part("Hinge arm")
        attributes = {
            "amount": 1, "tool": "Pliers", "detachment_time": 5,
            "non_destructive": True, "ageing": False,
        }
        spoiler["disassembly_parameters"]["assigned_connections"] = {
            "Hinge pin": {"Contact Bonding": attributes.copy()}
        }
        pin["disassembly_parameters"]["assigned_connections"] = {
            "Spoiler": {"Contact Bonding": attributes.copy()}
        }
        self.data = {"classification": None, "data": [{
            "name": "Analyzed assembly", "id": "root", "components": [
                spoiler,
                {"name": "Hinge", "id": "hinge", "classification": "Stale",
                 "components": [pin, arm]},
            ],
        }]}
        self.calc = CalculationDialog(self.data, "", self.config)
        self.addCleanup(self.calc.close)
        self.warning = self.enterContext(patch("tool.QMessageBox.warning"))
        self.critical = self.enterContext(patch("tool.QMessageBox.critical"))
        self.enterContext(patch("tool.QMessageBox.information"))

    def test_mixed_depth_parts_derive_assembly_and_decimal_reference_times(self):
        self.assertTrue(self.calc.calculate_classification_averages())
        self.assertEqual(self.data["classification"], "Tailgate")
        self.assertEqual(self.data["data"][0]["components"][1]["classification"], "Hinge assembly")
        self.assertEqual(self.calc.average_number_of_components, 28)
        self.assertEqual(self.calc.average_number_of_joining_connections, 93)
        self.assertEqual(self.calc.average_disassembly_time, 26520.5)
        self.assertEqual(self.calc.average_assembly_time, 7951.63)
        self.warning.assert_not_called()

    def test_stale_main_classification_is_recomputed(self):
        self.data["classification"] = "Spoiler"
        self.assertTrue(self.calc.calculate_classification_averages())
        self.assertEqual(self.data["classification"], "Tailgate")
        self.assertEqual(self.calc.average_number_of_components, 28)

    def test_repeated_part_type_uses_parent_reference(self):
        self.data["data"][0]["components"] = [part("Hinge pin"), part("Hinge pin")]
        self.assertTrue(self.calc.calculate_classification_averages())
        self.assertEqual(self.data["classification"], "Hinge assembly")
        self.assertEqual(self.calc.average_number_of_components, 2)

    def test_selection_dialog_uses_the_same_derivation(self):
        selection = SelectionDialog(copy.deepcopy(self.data), "", self.config)
        self.addCleanup(selection.close)
        selection.update_assembly_classification(selection.structure, self.hierarchy)
        self.assertEqual(selection.structure["classification"], "Tailgate")
        self.assertEqual(
            selection.structure["data"][0]["components"][1]["classification"],
            "Hinge assembly",
        )

    def test_complete_calculation_succeeds_with_derived_classification(self):
        self.calc.start_calculation()
        self.warning.assert_not_called()
        self.critical.assert_not_called()
        self.assertIn("All calculations completed successfully", self.calc.status_bar.text())
        self.assertEqual(self.data["classification"], "Tailgate")
        self.assertEqual(self.calc.number_of_components, 3)
        self.assertEqual(self.calc.all_parameters["parameter_number_of_components"], 1.0)
        self.assertIsInstance(self.calc.logistics_score, float)

    def test_missing_or_unknown_part_classification_stops_once(self):
        for classification in (None, "Unknown classification"):
            with self.subTest(classification=classification):
                self.warning.reset_mock()
                self.critical.reset_mock()
                self.data["data"][0]["components"][0]["basic_parameters"]["classification"] = classification
                with patch.object(self.calc, "calculate_number_of_components") as next_step:
                    self.calc.start_calculation()
                self.warning.assert_called_once()
                self.critical.assert_not_called()
                next_step.assert_not_called()
                self.assertIsNone(self.calc.average_number_of_components)
                self.assertIn("stopped", self.calc.status_bar.text().lower())

    def test_missing_reference_examples_stops_once(self):
        self.calc.classification_data["Vehicle"]["Body"]["Tailgate"].pop("Examples")
        with patch.object(self.calc, "calculate_number_of_components") as next_step:
            self.calc.start_calculation()
        self.warning.assert_called_once()
        self.critical.assert_not_called()
        next_step.assert_not_called()
        self.assertIsNone(self.calc.average_number_of_components)
        self.assertEqual(self.data["classification"], "Tailgate")

    def test_failed_rerun_clears_previous_averages_and_results(self):
        self.calc.start_calculation()
        self.assertTrue(self.calc.r_strategy_scores)
        self.data["data"][0]["components"][0]["basic_parameters"].pop("classification")
        self.calc.start_calculation()
        self.assertIsNone(self.calc.average_number_of_components)
        self.assertIsNone(self.calc.average_assembly_time)
        self.assertEqual(self.calc.all_parameters, {})
        self.assertEqual(self.calc.goal_scores, {})
        self.assertEqual(self.calc.r_strategy_scores, {})
        self.warning.assert_called_once()
        self.critical.assert_not_called()


if __name__ == "__main__":
    unittest.main()
