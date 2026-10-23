"""Derive assembly classifications from their leaf parts.

Copyright (C) 2026 Digiser, Riech & Nebel
SPDX-License-Identifier: GPL-3.0-only
"""

from collections import defaultdict
from collections.abc import Mapping
from typing import Any


def collect_part_classifications(assembly: Mapping[str, Any]) -> list[Any]:
    """Collect leaf classifications, retaining missing values as ``None``.

    Structural assemblies contribute their parts, not any stale classification
    stored on the assembly itself.
    """
    children = assembly.get("components") or []
    if children:
        return [
            classification
            for child in children
            for classification in collect_part_classifications(child)
        ]
    basic_parameters = assembly.get("basic_parameters") or {}
    return [basic_parameters.get("classification")]


def find_parent_classification(
    classification_structure: Mapping[str, Any], classifications: list[Any]
) -> str | None:
    """Return the deepest common strict ancestor of all part classifications.

    ``Examples`` contains reference data, not classification branches. Missing,
    unknown, or ambiguous classification names cannot establish an ancestor.
    Repeated instances of the same part type resolve to that type's parent.
    """
    if not classifications or any(
        not isinstance(name, str) or not name for name in classifications
    ):
        return None

    paths = defaultdict(list)

    def collect_paths(structure, ancestors=()):
        for name, children in structure.items():
            if name == "Examples" or not isinstance(children, Mapping):
                continue
            path = ancestors + (name,)
            paths[name].append(path)
            collect_paths(children, path)

    collect_paths(classification_structure)
    common_path = None
    for name in classifications:
        candidates = paths.get(name, [])
        if len(candidates) != 1:
            return None
        parent_path = candidates[0][:-1]
        if common_path is None:
            common_path = parent_path
        else:
            shared_length = 0
            for left, right in zip(common_path, parent_path):
                if left != right:
                    break
                shared_length += 1
            common_path = common_path[:shared_length]
        if not common_path:
            return None
    return common_path[-1]


def update_assembly_classifications(
    data: dict[str, Any], classification_structure: Mapping[str, Any]
) -> str | None:
    """Derive every structural assembly and update the document's main class.

    The first root is the main assembly, matching the project's data format.
    A root that is itself a part keeps its own assessed classification.
    """
    def update_node(node):
        children = node.get("components") or []
        if not children:
            return collect_part_classifications(node)
        classifications = []
        for child in children:
            classifications.extend(update_node(child))
        node["classification"] = find_parent_classification(
            classification_structure, classifications
        )
        return classifications

    roots = data.get("data") or []
    main_classification = None
    for index, root in enumerate(roots):
        classifications = update_node(root)
        if index == 0:
            main_classification = (
                root.get("classification")
                if root.get("components")
                else classifications[0]
            )
    data["classification"] = main_classification
    return main_classification
