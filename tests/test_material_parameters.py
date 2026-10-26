"""Keep material scores bounded while preserving mass-weighted assessments."""

from pathlib import Path
from types import SimpleNamespace
import unittest

from PyQt5.QtWidgets import QApplication

from tool import CalculationDialog


ROOT = Path(__file__).resolve().parents[1]


class MaterialParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        config = SimpleNamespace(
            materials_json=str(ROOT / "app_data/materials_validation.json"),
            parameter_to_goal_json=str(ROOT / "app_data/parameter_to_goal.json"),
            goal_to_r_strategy_json=str(ROOT / "app_data/goal_to_r_strategy.json"),
            joining_connections_json=str(ROOT / "joining_connections.json"),
            classification_json=str(ROOT / "app_data/classification_validation.json"),
        )
        self.calc = CalculationDialog({}, "", config)
        self.addCleanup(self.calc.close)

    def test_material_count_caps_seventeen_reference_groups_over_thirteen(self):
        # The supplied Tailgate classification produces 17 distinct reference
        # groups; 17 / 13 previously appeared as 1.31 in the parameter results.
        self.calc.material_list = [f"actual-{index}" for index in range(13)]
        self.calc.unique_component_materials = [
            {"materials": [f"reference-{index}"]} for index in range(17)
        ]
        self.assertEqual(self.calc.calculate_parameter_number_of_materials(), 1.0)
        self.assertEqual(self.calc.get_all_parameters()["parameter_number_of_materials"], 1.0)
        goals = self.calc.calculate_goal_achievement()
        self.assertEqual(goals["Reduction of material diversity"], 1.0)

    def test_material_count_preserves_fractional_scores_and_shared_materials(self):
        self.calc.material_list = ["a", "b", "c", "d"]
        self.calc.unique_component_materials = [
            {"materials": ["steel", "aluminum"]},
            {"materials": ["steel"]},
            {"materials": ["polymer"]},
            {"materials": ["polymer"]},
        ]
        self.assertEqual(self.calc.calculate_parameter_number_of_materials(), 0.5)

    def test_material_count_without_current_materials_is_unavailable(self):
        self.calc.material_list = []
        self.calc.unique_component_materials = [{"materials": ["steel"]}]
        self.calc.parameter_number_of_materials = 0.9
        self.assertIsNone(self.calc.calculate_parameter_number_of_materials())

    def test_all_six_binary_parameters_retain_mass_weighting(self):
        properties = (
            "recycling_criticality", "environmental_harmfulness",
            "health_harmfulness", "additives_or_fillers",
            "surface_coatings", "monomaterial",
        )
        self.calc.material_data = {
            f"material-{index}": {
                name: (index < 10 if name == "monomaterial" else index >= 10)
                for name in properties
            }
            for index in range(13)
        }
        # Ten materials qualify, but together they account for only 7% of mass.
        self.calc.data = {"data": [{"components": [
            {"material": f"material-{index}",
             "mass [kg]": 1 if index < 10 else 310 / 7,
             "components": []}
            for index in range(13)
        ]}]}
        self.calc.calculate_number_of_material_types()
        self.calc.calculate_parameters_material_binary()
        parameters = self.calc.get_all_parameters()
        for name in properties:
            with self.subTest(property=name):
                self.assertAlmostEqual(parameters[f"parameter_{name}"], 0.07)

        goals = self.calc.calculate_goal_achievement()
        self.assertAlmostEqual(goals["Selection of recyclable materials"], 0.07)
        self.assertAlmostEqual(goals["Avoidance of recycling-critical materials"], 0.07)


if __name__ == "__main__":
    unittest.main()
