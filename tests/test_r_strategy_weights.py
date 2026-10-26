"""Regression coverage for malformed goal-to-R-strategy weight rows."""

from copy import deepcopy
import json
import unittest

from r_strategy_weights import validate_goal_weights


class GoalWeightValidationTests(unittest.TestCase):
    def test_accepts_zero_negative_and_fractional_weights(self):
        weights = {"Goal": [0, 0.0, -4, 4.0, 1.172317705, -0.25, 1, 2, 3, 0]}
        self.assertIsNone(validate_goal_weights(weights))
        self.assertIsNone(validate_goal_weights({}))

    def test_rejects_non_object_roots(self):
        for root in (None, [], "{}", 42):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, "JSON object"):
                validate_goal_weights(root)

    def test_rows_must_be_lists_with_ten_positions(self):
        for row in (None, "0,0", {}, tuple([0] * 10)):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "Goal.*list.*10"):
                validate_goal_weights({"Goal": row})
        for size in (0, 9, 11):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, f"found {size}"):
                validate_goal_weights({"Goal": [0.0] * size})

    def test_decimal_comma_rows_are_reported_together(self):
        weights = json.loads('''{
            "Minimise disassembly time": [0,0,0,0,2.218838544,1.282473043,1,172317705,0,0.339962357,0],
            "Extend functional lifetime": [0,4,0,3.220104544,0,1,978375969,0,0,0,0]
        }''')
        with self.assertRaises(ValueError) as raised:
            validate_goal_weights(weights)
        message = str(raised.exception)
        for goal in weights:
            self.assertIn(goal, message)
        self.assertEqual(message.count("found 11"), 2)
        self.assertIn("decimal point", message)
        self.assertIn("comma creates an extra JSON value", message)

    def test_rejects_non_numeric_values_in_any_row(self):
        for value in (True, False, None, "0.0", [], {}):
            weights = {"Unused goal": [0.0] * 10}
            weights["Unused goal"][7] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Unused goal.*R7.*number"):
                validate_goal_weights(weights)

    def test_rejects_non_finite_and_out_of_range_weights(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                validate_goal_weights({"Goal": [value] + [0] * 9})
        for value in (-4.01, 4.01, 172317705, 10 ** 400):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "between -4.0 and 4.0"):
                validate_goal_weights({"Goal": [value] + [0] * 9})

    def test_rejects_invalid_goal_names(self):
        for goal in (None, 7, "", " "):
            with self.subTest(goal=goal), self.assertRaisesRegex(ValueError, "goal name"):
                validate_goal_weights({goal: [0.0] * 10})

    def test_validation_does_not_mutate_valid_or_invalid_data(self):
        valid = {"Goal": [0.0] * 10}
        invalid = {"Goal": [0.0] * 11}
        for weights in (valid, invalid):
            before = deepcopy(weights)
            try:
                validate_goal_weights(weights)
            except ValueError:
                pass
            self.assertEqual(weights, before)


if __name__ == "__main__":
    unittest.main()
