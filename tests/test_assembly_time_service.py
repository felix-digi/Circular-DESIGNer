"""Regression coverage for joining connections at every nesting level."""

import json
from pathlib import Path
import unittest

from assembly_time_service import AssemblyTimeService, compute_joining_assembly_time


CATALOGUE = Path(__file__).resolve().parents[1] / "joining_connections.json"


class AssemblyTimeServiceTests(unittest.TestCase):
    def test_catalogue_resolves_second_and_third_order_connections(self):
        service = AssemblyTimeService(CATALOGUE)
        for name, expected in (
            ("Contact Bonding", 7.56),
            ("Fastening through Expanding", 9.72),
            ("Tube Rolling", 9.72),
        ):
            with self.subTest(name=name):
                self.assertEqual(service.get_assembly_time(name), expected)

    def test_every_catalogue_name_is_indexed(self):
        catalogue = json.loads(CATALOGUE.read_text())
        service = AssemblyTimeService(catalogue)
        pending = list(catalogue)
        while pending:
            connection = pending.pop()
            with self.subTest(number=connection["number"]):
                self.assertIsNotNone(service.get_assembly_time(connection["name"]))
            pending.extend(connection.get("sub_connections", []))

    def test_nested_connections_contribute_once_per_mirrored_pair(self):
        connections = {
            "A": {"B": {"Contact Bonding": {"amount": 2}}},
            "B": {"A": {"Contact Bonding": {"amount": 2}}},
            "C": {"D": {"Fastening through Expanding": {"amount": 3}}},
            "D": {"C": {"Fastening through Expanding": {"amount": 3}}},
        }
        total, breakdown = compute_joining_assembly_time(
            connections, joining_connections=CATALOGUE, return_breakdown=True
        )
        self.assertAlmostEqual(total, 44.28)
        self.assertEqual(len(breakdown), 2)
        self.assertEqual(
            {item.connection_name: item.amount for item in breakdown},
            {"Contact Bonding": 2.0, "Fastening through Expanding": 3.0},
        )

    def test_explicit_times_do_not_stop_descendant_indexing(self):
        hierarchy = {
            "name": "Root", "assembly_time": 100,
            "sub_connections": [{
                "name": "Child", "assembly_time": 20,
                "sub_connections": [{
                    "name": "Grandchild", "assembly_time": 4,
                    "sub_connections": [{
                        "name": "Great-grandchild", "assembly_time": 2,
                    }],
                }],
            }],
        }
        self.assertEqual(AssemblyTimeService(hierarchy).assembly_time_by_name, {
            "Root": 100.0, "Child": 20.0,
            "Grandchild": 4.0, "Great-grandchild": 2.0,
        })

    def test_medians_use_direct_child_effective_times(self):
        hierarchy = {
            "name": "Root",
            "sub_connections": [
                {"name": "Branch", "sub_connections": [
                    {"name": "One", "assembly_time": 1},
                    {"name": "Two", "assembly_time": 2},
                    {"name": "Three", "assembly_time": 3},
                ]},
                {"name": "Other", "assembly_time": 100},
                {"name": "Unresolved"},
            ],
        }
        service = AssemblyTimeService(hierarchy)
        self.assertEqual(service.get_assembly_time("Branch"), 2.0)
        self.assertEqual(service.get_assembly_time("Root"), 51.0)
        self.assertIsNone(service.get_assembly_time("Unresolved"))

    def test_explicit_zero_remains_valid_with_children(self):
        service = AssemblyTimeService({
            "name": "Free", "assembly_time": 0,
            "sub_connections": [{"name": "Child", "assembly_time": 5}],
        })
        self.assertEqual(service.get_assembly_time("Free"), 0.0)
        self.assertEqual(service.get_assembly_time("Child"), 5.0)

    def test_parent_time_takes_precedence_for_duplicate_name(self):
        for explicit, expected in ((200, 200.0), (None, 6.0)):
            with self.subTest(explicit=explicit):
                service = AssemblyTimeService({
                    "name": "Duplicate", "assembly_time": explicit,
                    "sub_connections": [
                        {"name": "Duplicate", "assembly_time": 2},
                        {"name": "Other", "assembly_time": 10},
                    ],
                })
                self.assertEqual(service.get_assembly_time("Duplicate"), expected)
        self.assertEqual(AssemblyTimeService(CATALOGUE).get_assembly_time("Filling"), 201.12)

    def test_unresolved_names_keep_strict_and_non_strict_behavior(self):
        service = AssemblyTimeService([{"name": "Unresolved"}])
        connections = {"A": {"B": {"Unresolved": {"amount": 1}}}}
        with self.assertRaisesRegex(KeyError, "Unresolved"):
            service.compute_total(connections)
        self.assertEqual(service.compute_total(connections, strict=False), 0.0)
        self.assertEqual(service.get_assembly_time("Missing", default=7), 7)

    def test_invalid_descendant_time_is_validated_under_explicit_parent(self):
        with self.assertRaisesRegex(ValueError, "Invalid assembly_time.*Child"):
            AssemblyTimeService({
                "name": "Root", "assembly_time": 10,
                "sub_connections": [{"name": "Child", "assembly_time": "invalid"}],
            })


if __name__ == "__main__":
    unittest.main()
