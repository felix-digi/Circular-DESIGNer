""""    
    Assembly time calculation utilities.

    This module provides helpers for loading joining connection definitions,
    deriving effective assembly times, and computing total assembly time from an
    in-memory component connection mapping.
    
    Copyright (C) 2026  Digiser

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union


JsonLike = Union[str, Path, Mapping[str, Any], List[Any]]


@dataclass(frozen=True)
class ConnectionBreakdownItem:
    """
    Stores one deduplicated connection entry used in the assembly time total.

    Class-Level Constants:
        None.

    Attributes:
        part_a (str): The first part name in sorted order.
        part_b (str): The second part name in sorted order.
        connection_name (str): The joining connection assigned to the part pair.
        amount (float): The quantity of the connection between the two parts.
        assembly_time_per_unit (float): The effective assembly time for one unit of the connection.
        total_time (float): The computed contribution of the connection to the overall total.

    Methods:
        None. This dataclass is used as a structured data container.
    """
    part_a: str
    part_b: str
    connection_name: str
    amount: float
    assembly_time_per_unit: float
    total_time: float


class AssemblyTimeService:
    """
    Computes total assembly time from component connection data and joining definitions.

    Class-Level Constants:
        DEFAULT_JOINING_FILE (str): Default JSON filename used when no joining source is supplied.

    Attributes:
        _joining_file_path (Path): Absolute path to the sibling joining connections file.
        _joining_connections_source (Optional[JsonLike]): Original source used to load joining data.
        _joining_connections (Any): Parsed joining connection hierarchy.
        _assembly_time_by_name (Dict[str, float]): Lookup table mapping connection names to effective assembly times.

    Methods:
        __init__(joining_connections=None): Load joining connection data and build the lookup index.
        assembly_time_by_name(): Return a copy of the effective assembly time lookup.
        joining_file_path(): Return the resolved default joining file path.
        reload(joining_connections=None): Reload joining connection data and rebuild the lookup.
        get_assembly_time(connection_name, default=None): Look up an effective assembly time by name.
        compute_total(component_connections, return_breakdown=False, strict=True): Compute the total assembly time.
        _make_dedupe_key(part_a, part_b, connection_name): Create an order-independent deduplication key.
        _extract_amount(attributes, part_a, part_b, connection_name): Read and validate the connection amount.
        _build_assembly_time_index(joining_connections): Build the effective assembly time lookup table.
        _iter_mapping_items(mapping, root_name): Yield mapping items with validated input.
        _load_joining_connections(source): Load joining data from the configured source.
        _load_data(source): Parse joining data from memory or a file.
    """

    DEFAULT_JOINING_FILE = "joining_connections.json"

    def __init__(self, joining_connections: Optional[JsonLike] = None) -> None:
        """
        Initialize the assembly time service and build the effective time lookup.

        Args:
            joining_connections (Optional[JsonLike]): Optional in-memory data structure, JSON file path,
                or Python-literal file path containing joining connection definitions.

        Steps:
            1. Resolve the default sibling path for `joining_connections.json`.
            2. Store the original joining connection source.
            3. Load the joining connection hierarchy from the provided source or default file.
            4. Build the connection-name to effective-assembly-time lookup table.

        Returns:
            None: This constructor initializes the service state in place.

        Notes:
            - If `joining_connections` is omitted, the service loads the default file next to this module.
            - Effective assembly times can be explicit values or derived medians from direct child connections.
        """
        self._joining_file_path = Path(__file__).resolve().parent / self.DEFAULT_JOINING_FILE
        self._joining_connections_source = joining_connections
        self._joining_connections = self._load_joining_connections(joining_connections)
        self._assembly_time_by_name = self._build_assembly_time_index(self._joining_connections)

    @property
    def assembly_time_by_name(self) -> Dict[str, float]:
        """
        Return a copy of the effective assembly time lookup table.

        Args:
            None.

        Steps:
            1. Read the internally cached connection-name lookup table.
            2. Create a shallow copy of the lookup dictionary.
            3. Return the copied dictionary to the caller.

        Returns:
            Dict[str, float]: A copy of the connection-name to effective-assembly-time mapping.

        Notes:
            - A copy is returned so callers cannot mutate the service's internal cache directly.
        """
        return dict(self._assembly_time_by_name)

    @property
    def joining_file_path(self) -> Path:
        """
        Return the resolved path to the default joining connections file.

        Args:
            None.

        Steps:
            1. Read the cached default file path.
            2. Return the resolved `Path` object.

        Returns:
            Path: The default sibling path for `joining_connections.json`.

        Notes:
            - This path is resolved during service initialization.
        """
        return self._joining_file_path

    def reload(self, joining_connections: Optional[JsonLike] = None) -> None:
        """
        Reload joining connection data and rebuild the effective time lookup.

        Args:
            joining_connections (Optional[JsonLike]): Optional replacement source for joining
                connection definitions. If omitted, the last configured source is reused.

        Steps:
            1. Update the stored source when a new one is provided.
            2. Reload the joining connection hierarchy from the current source.
            3. Rebuild the effective assembly time lookup table from the loaded data.

        Returns:
            None: This method refreshes the cached data in place.

        Notes:
            - Passing `None` reloads from the previously stored source or the default sibling file.
        """
        if joining_connections is not None:
            self._joining_connections_source = joining_connections

        self._joining_connections = self._load_joining_connections(self._joining_connections_source)
        self._assembly_time_by_name = self._build_assembly_time_index(self._joining_connections)

    def get_assembly_time(self, connection_name: str, default: Optional[float] = None) -> Optional[float]:
        """
        Look up the effective assembly time for a connection name.

        Args:
            connection_name (str): The joining connection name to resolve.
            default (Optional[float]): Fallback value returned when the name is not present.

        Steps:
            1. Search the internal effective assembly time lookup using the given connection name.
            2. Return the matching value when found.
            3. Return the provided default value when the name is not available.

        Returns:
            Optional[float]: The effective assembly time for the connection, or the fallback value.

        Notes:
            - This method does not raise an error when the connection name is missing.
        """
        return self._assembly_time_by_name.get(connection_name, default)

    def compute_total(
        self,
        component_connections: Mapping[str, Any],
        *,
        return_breakdown: bool = False,
        strict: bool = True,
    ) -> Union[float, Tuple[float, List[ConnectionBreakdownItem]]]:
        """
        Compute total assembly time from an in-memory component connection mapping.

        Args:
            component_connections (Mapping[str, Any]): Mirrored component connection data where each
                real connection may appear from both participating parts.
            return_breakdown (bool): Whether to also return a per-connection breakdown list.
            strict (bool): Whether to raise an error for unresolved connection names instead of skipping them.

        Steps:
            1. Validate that `component_connections` is a mapping-like object.
            2. Iterate through each part pair and assigned connection entry.
            3. Deduplicate mirrored connections so each real connection is counted once.
            4. Resolve the effective assembly time for each connection name.
            5. Extract the connection amount and compute the total contribution.
            6. Accumulate the overall total and optionally store breakdown rows.
            7. Return either the numeric total or the total with breakdown items.

        Returns:
            Union[float, Tuple[float, List[ConnectionBreakdownItem]]]: The total assembly time, or a
            tuple containing the total and a list of breakdown items when `return_breakdown` is True.

        Notes:
            - Mirrored connections are deduplicated using an order-independent key.
            - When `strict` is False, unresolved connection names are ignored instead of raising `KeyError`.
            - Breakdown rows store part names in sorted order for stable output.
        """
        if not isinstance(component_connections, Mapping):
            raise TypeError(
                "component_connections must be an in-memory dictionary-like mapping."
            )

        total = 0.0
        breakdown: List[ConnectionBreakdownItem] = []
        seen_keys = set()

        for part_a, neighbors in self._iter_mapping_items(component_connections, root_name="component_connections"):
            if not isinstance(neighbors, Mapping):
                continue

            for part_b, connection_map in self._iter_mapping_items(
                neighbors,
                root_name=f"component_connections[{part_a!r}]",
            ):
                if not isinstance(connection_map, Mapping):
                    continue

                for connection_name, attributes in self._iter_mapping_items(
                    connection_map,
                    root_name=f"component_connections[{part_a!r}][{part_b!r}]",
                ):
                    dedupe_key = self._make_dedupe_key(part_a, part_b, connection_name)
                    if dedupe_key in seen_keys:
                        continue
                    seen_keys.add(dedupe_key)

                    if connection_name not in self._assembly_time_by_name:
                        if strict:
                            raise KeyError(
                                f"Connection name {connection_name!r} was not found in joining connections "
                                f"and no derived median assembly_time could be computed for it."
                            )
                        continue

                    amount = self._extract_amount(attributes, part_a, part_b, connection_name)
                    per_unit_time = float(self._assembly_time_by_name[connection_name])
                    connection_total = per_unit_time * amount
                    total += connection_total

                    if return_breakdown:
                        ordered_a, ordered_b = sorted((str(part_a), str(part_b)))
                        breakdown.append(
                            ConnectionBreakdownItem(
                                part_a=ordered_a,
                                part_b=ordered_b,
                                connection_name=str(connection_name),
                                amount=amount,
                                assembly_time_per_unit=per_unit_time,
                                total_time=connection_total,
                            )
                        )

        if return_breakdown:
            return total, breakdown
        return total

    @staticmethod
    def _make_dedupe_key(part_a: Any, part_b: Any, connection_name: Any) -> Tuple[str, str, str]:
        """
        Build an order-independent key for mirrored connection entries.

        Args:
            part_a (Any): The first part identifier.
            part_b (Any): The second part identifier.
            connection_name (Any): The connection type or name assigned to the part pair.

        Steps:
            1. Convert both part identifiers to strings.
            2. Sort the two part names to remove directionality.
            3. Convert the connection name to a string.
            4. Return the normalized tuple key.

        Returns:
            Tuple[str, str, str]: A deduplication key composed of the sorted part names and connection name.

        Notes:
            - This ensures `A -> B` and `B -> A` are treated as the same connection.
        """
        left, right = sorted((str(part_a), str(part_b)))
        return left, right, str(connection_name)

    @staticmethod
    def _extract_amount(attributes: Any, part_a: Any, part_b: Any, connection_name: Any) -> float:
        """
        Extract and validate the numeric `amount` for a connection entry.

        Args:
            attributes (Any): Connection attribute mapping expected to contain an `amount` field.
            part_a (Any): The first part identifier for error reporting.
            part_b (Any): The second part identifier for error reporting.
            connection_name (Any): The connection name for error reporting.

        Steps:
            1. Validate that `attributes` is mapping-like.
            2. Check whether the `amount` field is present.
            3. Convert the amount value to a floating-point number.
            4. Raise a descriptive error when validation or conversion fails.
            5. Return the validated numeric amount.

        Returns:
            float: The connection amount as a floating-point number.

        Notes:
            - A `KeyError` is raised when `amount` is missing.
            - A `ValueError` is raised when the amount cannot be parsed as numeric.
        """
        if not isinstance(attributes, Mapping):
            raise TypeError(
                f"Expected attributes for connection {connection_name!r} between "
                f"{part_a!r} and {part_b!r} to be a mapping, got {type(attributes).__name__}."
            )

        if "amount" not in attributes:
            raise KeyError(
                f"Missing 'amount' for connection {connection_name!r} between {part_a!r} and {part_b!r}."
            )

        amount = attributes["amount"]
        try:
            return float(amount)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid 'amount' value {amount!r} for connection {connection_name!r} "
                f"between {part_a!r} and {part_b!r}."
            ) from exc

    @classmethod
    def _build_assembly_time_index(cls, joining_connections: Any) -> Dict[str, float]:
        """
        Build a lookup table of connection names to effective assembly times.

        Args:
            joining_connections (Any): Parsed joining connection hierarchy as a mapping or list.

        Steps:
            1. Traverse every child recursively, even when its parent has an explicit time.
            2. Use the explicit `assembly_time` value when a node defines one.
            3. Otherwise inspect direct `sub_connections` for resolvable child times.
            4. Derive the node's effective time as the median of direct child times.
            5. Store each resolvable node in the lookup dictionary.
            6. Return the completed lookup table.

        Returns:
            Dict[str, float]: A mapping of connection names to explicit or derived assembly times.

        Notes:
            - Nodes without an explicit time and without resolvable child times are omitted.
            - Derived values use the median of direct child effective times.
            - A node's explicit time takes precedence over its child-derived median.
            - Parents are indexed after descendants, preserving parent values for repeated names.
            - Invalid `assembly_time` values raise `ValueError`.
        """
        index: Dict[str, float] = {}

        def effective_time(node: Any) -> Optional[float]:
            if not isinstance(node, Mapping):
                return None

            name = node.get("name")
            raw_time = node.get("assembly_time", None)

            # Index all descendants before resolving this node's own time.
            child_times: List[float] = []
            children = node.get("sub_connections")
            if isinstance(children, list):
                for child in children:
                    child_time = effective_time(child)
                    if child_time is not None:
                        child_times.append(float(child_time))

            if raw_time is not None:
                try:
                    value = float(raw_time)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid assembly_time for connection {name!r}: {raw_time!r}"
                    ) from exc
                if name is not None:
                    index[str(name)] = value
                return value

            if not child_times:
                return None

            derived = float(median(child_times))
            if name is not None:
                index[str(name)] = derived
            return derived

        if isinstance(joining_connections, list):
            for item in joining_connections:
                effective_time(item)
        elif isinstance(joining_connections, Mapping):
            effective_time(joining_connections)
        else:
            raise TypeError(
                "joining_connections must be a list or dictionary-like structure."
            )

        return index

    @staticmethod
    def _iter_mapping_items(mapping: Mapping[Any, Any], *, root_name: str) -> Iterable[Tuple[Any, Any]]:
        """
        Yield items from a mapping after validating the input type.

        Args:
            mapping (Mapping[Any, Any]): Mapping object whose items should be returned.
            root_name (str): Human-readable name used in validation error messages.

        Steps:
            1. Validate that `mapping` implements the mapping interface.
            2. Raise a descriptive `TypeError` when validation fails.
            3. Return the mapping's item iterator.

        Returns:
            Iterable[Tuple[Any, Any]]: The items of the mapping.

        Notes:
            - This helper improves error messages for nested traversal operations.
        """
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{root_name} must be a mapping, got {type(mapping).__name__}.")
        return mapping.items()

    def _load_joining_connections(self, source: Optional[JsonLike]) -> Any:
        """
        Load joining connection data from the configured source.

        Args:
            source (Optional[JsonLike]): Optional in-memory data or file path to load.

        Steps:
            1. Replace a missing source with the default sibling joining file path.
            2. Delegate parsing to `_load_data`.
            3. Return the parsed joining connection structure.

        Returns:
            Any: The parsed joining connection data.

        Notes:
            - When `source` is `None`, `joining_connections.json` next to this file is used.
        """
        if source is None:
            source = self._joining_file_path
        return self._load_data(source)

    @classmethod
    def _load_data(cls, source: JsonLike) -> Any:
        """
        Load and parse joining connection data from memory or a file.

        Args:
            source (JsonLike): In-memory mapping/list, JSON file path, or Python-literal file path.

        Steps:
            1. Return the value directly when it is already a mapping or list.
            2. Read the file content when a path-like source is provided.
            3. Attempt to parse the file as JSON.
            4. Fall back to `ast.literal_eval` when JSON parsing fails.
            5. Raise a descriptive error when neither format can be parsed.

        Returns:
            Any: The parsed data structure loaded from the provided source.

        Notes:
            - Python-literal files may use single quotes and boolean literals.
            - Inline comments are not supported in files loaded by this helper.
        """
        if isinstance(source, (Mapping, list)):
            return source

        path = Path(source)
        text = path.read_text(encoding="utf-8").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Could not parse file {str(path)!r} as JSON or Python literal."
            ) from exc


def compute_joining_assembly_time(
    component_connections: Mapping[str, Any],
    *,
    joining_connections: Optional[JsonLike] = None,
    return_breakdown: bool = False,
    strict: bool = True,
) -> Union[float, Tuple[float, List[ConnectionBreakdownItem]]]:
    """
    Compute total assembly time with a one-call convenience wrapper.

    Args:
        component_connections (Mapping[str, Any]): In-memory mirrored component connection mapping.
        joining_connections (Optional[JsonLike]): Optional joining connection source or file path.
        return_breakdown (bool): Whether to also return breakdown rows.
        strict (bool): Whether to raise an error for unresolved connection names.

    Steps:
        1. Create an `AssemblyTimeService` using the provided joining connection source.
        2. Call `compute_total` with the supplied component connection data.
        3. Return the computed result to the caller.

    Returns:
        Union[float, Tuple[float, List[ConnectionBreakdownItem]]]: The total assembly time, or the
        total with breakdown items when `return_breakdown` is True.

    Notes:
        - By default, the joining connection file is loaded from the same folder as this module.
        - This function is a thin wrapper around `AssemblyTimeService.compute_total`.
    """
    service = AssemblyTimeService(joining_connections)
    return service.compute_total(
        component_connections,
        return_breakdown=return_breakdown,
        strict=strict,
    )
