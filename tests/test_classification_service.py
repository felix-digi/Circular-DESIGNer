"""Regression coverage for classification inheritance across nested parts."""

from copy import deepcopy
import unittest

from classification_service import (
    collect_part_classifications,
    find_parent_classification,
    update_assembly_classifications,
)


CLASSIFICATION = {
    "Vehicle": {"Body": {
        "Tailgate": {
            "Examples": [{"name": "Reference tailgate"}],
            "Spoiler": {"Examples": []},
            "Hinge Assembly": {
                "Hinge pin": {"Examples": []},
                "Hinge arm": {"Examples": []},
            },
            "Emblem Assembly": {
                "Emblem": {"Examples": []},
                "Emblem Carrier": {"Examples": []},
            },
        },
        "Hood": {"Hood sheet": {"Examples": []}},
    }},
}


def part(classification):
    return {"basic_parameters": {"classification": classification}, "components": []}


class ClassificationServiceTests(unittest.TestCase):
    def test_mixed_depth_parts_derive_tailgate(self):
        self.assertEqual(find_parent_classification(
            CLASSIFICATION, ["Spoiler", "Hinge pin", "Emblem Carrier"]
        ), "Tailgate")

    def test_sibling_subset_uses_deepest_parent(self):
        self.assertEqual(find_parent_classification(
            CLASSIFICATION, ["Hinge pin", "Hinge arm"]
        ), "Hinge Assembly")
        self.assertEqual(find_parent_classification(
            CLASSIFICATION, ["Hinge pin", "Emblem"]
        ), "Tailgate")

    def test_repeated_type_and_single_part_use_strict_parent(self):
        for classifications in (["Hinge pin"], ["Hinge pin", "Hinge pin"]):
            with self.subTest(classifications=classifications):
                self.assertEqual(find_parent_classification(
                    CLASSIFICATION, classifications
                ), "Hinge Assembly")

    def test_missing_unknown_and_empty_classifications_have_no_parent(self):
        for classifications in ([], [None], [""], ["Unknown"],
                                ["Spoiler", None], ["Spoiler", "Unknown"],
                                ["Vehicle"], [["Spoiler"]]):
            with self.subTest(classifications=classifications):
                self.assertIsNone(find_parent_classification(CLASSIFICATION, classifications))

    def test_examples_are_not_classification_nodes(self):
        hierarchy = deepcopy(CLASSIFICATION)
        hierarchy["Vehicle"]["Body"]["Tailgate"]["Examples"] = {
            "False classification": {"Spoiler": {}},
        }
        self.assertEqual(find_parent_classification(hierarchy, ["Spoiler"]), "Tailgate")
        self.assertIsNone(find_parent_classification(hierarchy, ["False classification"]))

    def test_ambiguous_names_do_not_choose_an_arbitrary_branch(self):
        hierarchy = {"Left": {"Part": {}}, "Right": {"Part": {}}}
        self.assertIsNone(find_parent_classification(hierarchy, ["Part"]))
        self.assertIsNone(find_parent_classification(hierarchy, ["Left", "Right"]))

    def test_nested_assemblies_are_derived_from_leaves_not_stale_values(self):
        hinge = {"classification": "Hood", "basic_parameters": {"classification": "Hood"},
                 "components": [part("Hinge pin"), part("Hinge arm")]}
        root = {"classification": "Hood", "components": [hinge, part("Spoiler")]}
        data = {"classification": "Hood", "data": [root]}
        before_parts = deepcopy(hinge["components"])
        self.assertEqual(collect_part_classifications(root), ["Hinge pin", "Hinge arm", "Spoiler"])
        self.assertEqual(update_assembly_classifications(data, CLASSIFICATION), "Tailgate")
        self.assertEqual(data["classification"], "Tailgate")
        self.assertEqual(root["classification"], "Tailgate")
        self.assertEqual(hinge["classification"], "Hinge Assembly")
        self.assertEqual(hinge["components"], before_parts)

    def test_unclassified_leaf_clears_stale_assembly_classification(self):
        root = {"classification": "Tailgate", "components": [part("Spoiler"), {}]}
        data = {"classification": "Tailgate", "data": [root]}
        self.assertEqual(collect_part_classifications(root), ["Spoiler", None])
        self.assertIsNone(update_assembly_classifications(data, CLASSIFICATION))
        self.assertIsNone(root["classification"])
        self.assertIsNone(data["classification"])

    def test_each_root_is_updated_but_first_root_remains_main(self):
        roots = [{"components": [part("Spoiler")]},
                 {"components": [part("Hood sheet")]}]
        data = {"data": roots}
        self.assertEqual(update_assembly_classifications(data, CLASSIFICATION), "Tailgate")
        self.assertEqual(roots[1]["classification"], "Hood")

    def test_empty_document_and_root_part(self):
        for data in ({}, {"data": []}, {"data": None, "classification": "Stale"}):
            with self.subTest(data=data):
                self.assertIsNone(update_assembly_classifications(data, CLASSIFICATION))
                self.assertIsNone(data["classification"])
        data = {"data": [part("Spoiler")]}
        self.assertEqual(update_assembly_classifications(data, CLASSIFICATION), "Spoiler")


if __name__ == "__main__":
    unittest.main()
