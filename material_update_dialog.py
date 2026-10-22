""""    
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

import json
import os
import tempfile
from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QCompleter,
    QWidget,
    QHBoxLayout
)


BOOLEAN_FIELDS = [
    ("recycling_criticality", "Recycling criticality"),
    ("incineration_for_energy_recovery", "Incineration for energy recovery"),
    ("environmental_harmfulness", "Environmental harmfulness"),
    ("health_harmfulness", "Health harmfulness"),
    ("monomaterial", "Monomaterial"),
    ("additives_or_fillers", "Additives or fillers"),
    ("surface_coatings", "Surface coatings"),
]

AVAILABILITY_OPTIONS = [
    "Regionally",
    "Nationally",
    "Same continent",
    "Other continent",
]


def load_material_file(json_path: str) -> Dict[str, Any]:
    """
    Load and validate the material JSON file.

    Args:
        json_path (str): Path to the JSON file containing the top-level `materials` object.

    Steps:
        1. Open the JSON file using UTF-8 encoding.
        2. Parse the file contents with `json.load`.
        3. Validate that the parsed value is a dictionary.
        4. Validate that the dictionary contains a top-level `materials` mapping.
        5. Return the validated data structure.

    Returns:
        Dict[str, Any]: Parsed material data containing a top-level `materials` dictionary.

    Notes:
        - Raises `ValueError` when the JSON structure does not match the expected schema.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict) or "materials" not in data or not isinstance(data["materials"], dict):
        raise ValueError("JSON must contain a top-level 'materials' object.")

    return data


def save_material_file_atomic(json_path: str, data: Dict[str, Any]) -> None:
    """
    Save material data to disk using an atomic replace workflow.

    Args:
        json_path (str): Destination JSON file path.
        data (Dict[str, Any]): Material data to write.

    Steps:
        1. Resolve the destination directory for the JSON file.
        2. Create a temporary file in the same directory.
        3. Write the JSON content with indentation and UTF-8 encoding.
        4. Replace the original file with the temporary file using `os.replace`.
        5. Remove the temporary file if an exception occurs before replacement.

    Returns:
        None: This function writes the JSON file in place.

    Notes:
        - Writing to a temporary file first reduces the risk of leaving a partially written output file.
    """
    directory = os.path.dirname(os.path.abspath(json_path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix="materials_", suffix=".tmp", dir=directory)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_path, json_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def empty_material_record() -> Dict[str, Any]:
    """
    Create a blank material record with all expected fields present.

    Args:
        None.

    Steps:
        1. Build a dictionary containing all supported material fields.
        2. Initialize each field with `None`.
        3. Return the empty material record.

    Returns:
        Dict[str, Any]: Empty material record ready to be inserted into the material store.

    Notes:
        - This record is used when a new material is created from the import workflow.
    """
    return {
        "recycling_percentage": None,
        "recyclate_content": None,
        "recycling_criticality": None,
        "incineration_for_energy_recovery": None,
        "environmental_harmfulness": None,
        "health_harmfulness": None,
        "monomaterial": None,
        "additives_or_fillers": None,
        "surface_coatings": None,
        "availability": None,
        "density [kg/mm^3]": None,
    }


def number_to_text(value: Any) -> str:
    """
    Convert a stored value into display text for a line edit widget.

    Args:
        value (Any): Value to display in the user interface.

    Steps:
        1. Return an empty string when the input value is `None`.
        2. Format floating-point values without unnecessary trailing zeros.
        3. Convert all other values to strings.
        4. Return the generated text.

    Returns:
        str: Display-ready text representation of the input value.

    Notes:
        - This helper keeps numeric text fields compact when existing values are loaded into the dialog.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def normalize_number(value: float):
    """
    Normalize a numeric value so whole numbers are stored as integers.

    Args:
        value (float): Numeric value to normalize.

    Steps:
        1. Check whether the float represents an integer value.
        2. Return an integer when the value has no fractional component.
        3. Otherwise return the original floating-point value.

    Returns:
        int | float: Integer value for whole numbers, otherwise the original float.

    Notes:
        - This keeps percentage fields compact when users enter values such as `50.0`.
    """
    return int(value) if float(value).is_integer() else float(value)


class MaterialSelectorDialog(QDialog):
    """
    Searchable dialog for selecting an existing material to update.

    Class-Level Constants:
        None.

    Attributes:
        _selected_name (Optional[str]): Material name chosen by the user after validation.
        _name_lookup (Dict[str, str]): Case-insensitive lookup from typed text to canonical material names.
        combo (QComboBox): Editable combo box used for searching and selecting materials.
        select_button (QPushButton): Button that validates and accepts the selected material.

    Methods:
        __init__(material_names, parent=None): Build the searchable selector dialog.
        accept_selection(): Validate the current text and accept the dialog when a known material is selected.
        selected_name(): Return the validated selected material name.
    """

    def __init__(self, material_names, parent=None):
        """
        Initialize the searchable material selector dialog.

        Args:
            material_names (Iterable[str]): Existing material names that can be selected.
            parent (Optional[QWidget]): Parent widget for the dialog.

        Steps:
            1. Initialize the base dialog and configure the title, modality, and size.
            2. Build a case-insensitive lookup table for all material names.
            3. Create an editable combo box populated with the available materials.
            4. Attach a completer for case-insensitive search and popup suggestions.
            5. Add dialog buttons for selection and cancellation.
            6. Connect button and keyboard events to the selection handler.

        Returns:
            None: This constructor builds the dialog UI in place.

        Notes:
            - This dialog is intended only for manually selecting an existing material to update.
        """
        super().__init__(parent)
        self.setWindowTitle("Select material to update")
        self.setModal(True)
        self.resize(420, 120)

        self._selected_name: Optional[str] = None
        self._name_lookup = {name.casefold(): name for name in material_names}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Search and select an existing material:"))

        self.combo = QComboBox()
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QComboBox.NoInsert)
        self.combo.addItems(material_names)

        completer = QCompleter(material_names, self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        try:
            completer.setFilterMode(Qt.MatchContains)
        except AttributeError:
            pass

        self.combo.setCompleter(completer)
        layout.addWidget(self.combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.select_button = QPushButton("Select")
        buttons.addButton(self.select_button, QDialogButtonBox.AcceptRole)

        self.select_button.clicked.connect(self.accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.combo.lineEdit().returnPressed.connect(self.accept_selection)
        self.combo.lineEdit().selectAll()

    def accept_selection(self):
        """
        Validate the typed material name and accept the dialog when it matches an existing entry.

        Args:
            None.

        Steps:
            1. Read and trim the current text from the editable combo box.
            2. Warn the user when no value was entered.
            3. Resolve the typed value through the case-insensitive name lookup.
            4. Warn the user when the value does not match an existing material.
            5. Store the canonical material name and accept the dialog.

        Returns:
            None: This method updates dialog state and may close the dialog.

        Notes:
            - Only existing materials can be selected in this workflow.
        """
        typed = self.combo.currentText().strip()
        if not typed:
            QMessageBox.warning(self, "No selection", "Please choose a material.")
            return

        selected = self._name_lookup.get(typed.casefold())
        if selected is None:
            QMessageBox.warning(
                self,
                "Unknown material",
                "Please select an existing material from the list. This dialog only updates materials."
            )
            return

        self._selected_name = selected
        self.accept()

    @property
    def selected_name(self) -> Optional[str]:
        """
        Return the validated material name selected in the dialog.

        Args:
            None.

        Steps:
            1. Read the internally stored selected material name.
            2. Return the selected value or `None`.

        Returns:
            Optional[str]: Canonical material name chosen by the user, or `None` when nothing was accepted.

        Notes:
            - The value is set only after `accept_selection` validates the current input.
        """
        return self._selected_name


class MaterialUpdateDialog(QDialog):
    """
    Dialog for editing and saving a single material record.

    Class-Level Constants:
        None.

    Attributes:
        json_path (str): Path to the material JSON file being edited.
        data (Dict[str, Any]): Loaded material data containing the full `materials` mapping.
        material_name (str): Name of the material currently being edited.
        material (Dict[str, Any]): Dictionary containing the editable values for the selected material.
        lookup_country_alpha2 (Optional[str]): Optional country hint shown beside the availability field.
        recycling_percentage_edit (QLineEdit): Input field for recycling percentage.
        recyclate_content_edit (QLineEdit): Input field for recyclate content.
        density_edit (QLineEdit): Input field for material density.
        bool_widgets (Dict[str, QComboBox]): Combo boxes for boolean material attributes.
        availability_combo (QComboBox): Combo box for availability selection.
        update_button (QPushButton): Button that validates and saves the edited material.

    Methods:
        __init__(json_path, data, material_name, parent=None, lookup_country_alpha2=None): Build the editor dialog.
        _set_combo_value(combo, value): Select a combo box entry by its stored data value.
        _parse_required_float(widget, field_name): Parse a required numeric input field.
        _collect_values(): Validate and collect all edited material values.
        handle_update(): Save validated changes and close the dialog on success.
    """

    def __init__(
        self,
        json_path: str,
        data: Dict[str, Any],
        material_name: str,
        parent=None,
        lookup_country_alpha2: Optional[str] = None,
    ):
        """
        Initialize the material update dialog for one selected material.

        Args:
            json_path (str): Path to the material JSON file being edited.
            data (Dict[str, Any]): Loaded JSON payload containing the `materials` mapping.
            material_name (str): Name of the material to update.
            parent (Optional[QWidget]): Parent widget for the dialog.
            lookup_country_alpha2 (Optional[str]): Optional country code displayed as a hint near availability.

        Steps:
            1. Initialize the base dialog and configure the title, modality, and size.
            2. Store the file path, loaded data, selected material name, and optional country hint.
            3. Create editable numeric fields for percentage and density values.
            4. Build yes/no combo boxes for all configured boolean material fields.
            5. Build the availability combo box and preselect the current value.
            6. Add an optional lookup country label beside the availability control.
            7. Add the update and cancel buttons and connect their handlers.

        Returns:
            None: This constructor builds the dialog UI in place.

        Notes:
            - Existing material values are preloaded into the form.
            - Availability values not present in the default options are preserved and shown.
        """
        super().__init__(parent)
        self.setWindowTitle(f"Update material: {material_name}")
        self.setModal(True)
        self.resize(620, 540)

        self.json_path = json_path
        self.data = data
        self.material_name = material_name
        self.material = self.data["materials"][material_name]
        self.lookup_country_alpha2 = lookup_country_alpha2

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel(f"<b>Material:</b> {material_name}"))

        form = QFormLayout()
        outer.addLayout(form)

        self.recycling_percentage_edit = QLineEdit(number_to_text(self.material.get("recycling_percentage")))
        self.recyclate_content_edit = QLineEdit(number_to_text(self.material.get("recyclate_content")))
        self.density_edit = QLineEdit(number_to_text(self.material.get("density [kg/mm^3]")))

        form.addRow("Recycling percentage (%)", self.recycling_percentage_edit)
        form.addRow("Recyclate content (%)", self.recyclate_content_edit)

        self.bool_widgets = {}
        for key, label in BOOLEAN_FIELDS:
            combo = QComboBox()
            combo.addItem("Select...", None)
            combo.addItem("Yes", True)
            combo.addItem("No", False)

            current_value = self.material.get(key, None)
            if current_value is True:
                combo.setCurrentIndex(1)
            elif current_value is False:
                combo.setCurrentIndex(2)
            else:
                combo.setCurrentIndex(0)

            self.bool_widgets[key] = combo
            form.addRow(label, combo)

        self.availability_combo = QComboBox()
        self.availability_combo.addItem("Select availability...", None)
        for option in AVAILABILITY_OPTIONS:
            self.availability_combo.addItem(option, option)

        current_availability = self.material.get("availability")
        if current_availability and current_availability not in AVAILABILITY_OPTIONS:
            self.availability_combo.addItem(current_availability, current_availability)

        self._set_combo_value(self.availability_combo, current_availability)

        availability_widget = QWidget()
        availability_layout = QHBoxLayout(availability_widget)
        availability_layout.setContentsMargins(0, 0, 0, 0)
        availability_layout.setSpacing(8)

        if self.lookup_country_alpha2:
            self.country_hint_label = QLabel(f"Lookup country: {self.lookup_country_alpha2}")
            availability_layout.addWidget(self.country_hint_label, 0)

        availability_layout.addWidget(self.availability_combo, 1)

        form.addRow("Availability:", availability_widget)

        form.addRow("Density [kg/mm^3]", self.density_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.update_button = QPushButton("Update")
        buttons.addButton(self.update_button, QDialogButtonBox.AcceptRole)

        self.update_button.clicked.connect(self.handle_update)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: Any):
        """
        Select the first combo box entry whose stored item data matches the given value.

        Args:
            combo (QComboBox): Combo box to update.
            value (Any): Item data value to search for.

        Steps:
            1. Iterate through all combo box items.
            2. Compare each item's stored data with the requested value.
            3. Set the matching index when a match is found.
            4. Fall back to the first entry when no match exists.

        Returns:
            None: This method updates the combo box selection in place.

        Notes:
            - This helper is used to restore saved availability values.
        """
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return
        combo.setCurrentIndex(0)

    def _parse_required_float(self, widget: QLineEdit, field_name: str) -> float:
        """
        Parse a required numeric value from a line edit widget.

        Args:
            widget (QLineEdit): Text input widget containing the numeric value.
            field_name (str): Human-readable field name used in validation messages.

        Steps:
            1. Read and trim the current widget text.
            2. Raise an error when the field is empty.
            3. Normalize commas to decimal points.
            4. Convert the text to a floating-point number.
            5. Raise a descriptive error when conversion fails.

        Returns:
            float: Parsed numeric value from the widget.

        Notes:
            - This helper is used for recycling percentage, recyclate content, and density inputs.
        """
        raw = widget.text().strip()
        if not raw:
            raise ValueError(f'"{field_name}" is required.')

        raw = raw.replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f'"{field_name}" must be a valid number.')

    def _collect_values(self) -> Dict[str, Any]:
        """
        Validate all form inputs and collect them into a material value dictionary.

        Args:
            None.

        Steps:
            1. Parse the required numeric input fields.
            2. Validate percentage ranges and positive density.
            3. Validate that an availability option has been selected.
            4. Normalize numeric values for storage.
            5. Validate that every boolean field has a yes or no answer.
            6. Build and return the updated material value dictionary.

        Returns:
            Dict[str, Any]: Validated material values ready to be merged into the data store.

        Notes:
            - Raises `ValueError` whenever a required input is missing or invalid.
        """
        recycling_percentage = self._parse_required_float(
            self.recycling_percentage_edit, "Recycling percentage"
        )
        recyclate_content = self._parse_required_float(
            self.recyclate_content_edit, "Recyclate content"
        )
        density = self._parse_required_float(self.density_edit, "Density")

        if not (0 <= recycling_percentage <= 100):
            raise ValueError('"Recycling percentage" must be between 0 and 100.')
        if not (0 <= recyclate_content <= 100):
            raise ValueError('"Recyclate content" must be between 0 and 100.')
        if density <= 0:
            raise ValueError('"Density" must be greater than 0.')

        availability = self.availability_combo.currentData()
        if availability is None:
            raise ValueError('"Availability" must be selected.')

        values = {
            "recycling_percentage": normalize_number(recycling_percentage),
            "recyclate_content": normalize_number(recyclate_content),
            "availability": availability,
            "density [kg/mm^3]": float(density),
        }

        for key, label in BOOLEAN_FIELDS:
            value = self.bool_widgets[key].currentData()
            if value is None:
                raise ValueError(f'"{label}" must be answered with Yes or No.')
            values[key] = value

        return values

    def handle_update(self):
        """
        Validate the form, save the updated material, and close the dialog on success.

        Args:
            None.

        Steps:
            1. Collect and validate the edited field values.
            2. Show a warning dialog when validation fails.
            3. Merge the validated values into the selected material record.
            4. Save the updated material file atomically.
            5. Show an error dialog when saving fails.
            6. Show a success message and accept the dialog when saving succeeds.

        Returns:
            None: This method updates persistent data and dialog state.

        Notes:
            - The dialog is accepted only after a successful save operation.
        """
        try:
            updated_values = self._collect_values()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid input", str(exc))
            return

        self.data["materials"][self.material_name].update(updated_values)

        try:
            save_material_file_atomic(self.json_path, self.data)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Could not overwrite the JSON file:\n{exc}")
            return

        QMessageBox.information(
            self,
            "Material updated",
            f'"{self.material_name}" was updated successfully.'
        )
        self.accept()


def open_material_update_dialog(parent, json_path: str) -> bool:
    """
    Launch the manual material update workflow for an existing material.

    Args:
        parent (QWidget): Parent widget for dialogs and message boxes.
        json_path (str): Path to the material JSON file.

    Steps:
        1. Load and validate the material file.
        2. Show an error dialog and return False when loading fails.
        3. Build a sorted list of available material names.
        4. Show an informational dialog and return False when no materials exist.
        5. Open the searchable material selector dialog.
        6. Return False when the user cancels selection.
        7. Open the material editor dialog for the selected material.
        8. Return whether the editor dialog was accepted.

    Returns:
        bool: True when the update dialog is completed successfully, otherwise False.

    Notes:
        - This workflow updates only existing materials and does not create new ones.
    """
    try:
        data = load_material_file(json_path)
    except Exception as exc:
        QMessageBox.critical(parent, "Load failed", f"Could not read JSON file:\n{exc}")
        return False

    material_names = sorted(data["materials"].keys(), key=str.casefold)
    if not material_names:
        QMessageBox.information(parent, "No materials", "No materials were found in the JSON file.")
        return False

    selector = MaterialSelectorDialog(material_names, parent)
    if selector.exec_() != QDialog.Accepted or not selector.selected_name:
        return False

    editor = MaterialUpdateDialog(json_path, data, selector.selected_name, parent)
    return editor.exec_() == QDialog.Accepted


def open_or_create_material_update_from_import(
    parent,
    json_path: str,
    material_name: str,
    lookup_country_alpha2: Optional[str] = None,
) -> bool:
    """
    Open the material editor from an import workflow, creating the material when needed.

    Args:
        parent (QWidget): Parent widget for dialogs and message boxes.
        json_path (str): Path to the material JSON file.
        material_name (str): Material name to edit or create.
        lookup_country_alpha2 (Optional[str]): Optional country hint to show beside availability.

    Steps:
        1. Trim the provided material name and validate that it is not empty.
        2. Load and validate the material file.
        3. Show an error dialog and return False when loading fails.
        4. Create an empty material record when the material does not yet exist.
        5. Save the file immediately after adding a new material.
        6. Reload the updated file content.
        7. Open the material editor dialog for the target material.
        8. Return whether the editor dialog was accepted.

    Returns:
        bool: True when the dialog is completed successfully, otherwise False.

    Notes:
        - This workflow can create a new material before opening the editor.
    """
    material_name = material_name.strip()
    if not material_name:
        QMessageBox.warning(parent, "Missing material", "No material name was provided.")
        return False

    try:
        data = load_material_file(json_path)
    except Exception as exc:
        QMessageBox.critical(parent, "Load failed", f"Could not read JSON file:\n{exc}")
        return False

    materials = data["materials"]

    if material_name not in materials:
        materials[material_name] = empty_material_record()
        try:
            save_material_file_atomic(json_path, data)
        except Exception as exc:
            QMessageBox.critical(parent, "Save failed", f"Could not add the new material:\n{exc}")
            return False

        data = load_material_file(json_path)

    editor = MaterialUpdateDialog(
        json_path=json_path,
        data=data,
        material_name=material_name,
        parent=parent,
        lookup_country_alpha2=lookup_country_alpha2,
    )
    return editor.exec_() == QDialog.Accepted


def _extract_country_from_single_material(material_dict: Dict[str, Any]) -> Optional[str]:
    """
    Extract the countryAlpha2 value from one lookup material record.

    Args:
        material_dict (Dict[str, Any]): Material lookup dictionary that may contain nested address data.

    Steps:
        1. Traverse the nested lookup structure toward `countryAlpha2`.
        2. Return None when the value is missing.
        3. Convert the value to a stripped string.
        4. Return the normalized value or None when the final string is empty.

    Returns:
        Optional[str]: Two-letter country code when available, otherwise None.

    Notes:
        - This helper expects the nested lookup format used by the import material lookup data.
    """
    value = (
        material_dict
        .get("raw", {})
        .get("bpdm_address", {})
        .get("physicalPostalAddress", {})
        .get("countryAlpha2")
    )
    if value is None:
        return None

    value = str(value).strip()
    return value or None


def extract_country_alpha2_from_lookup_material(
    lookup_data: Dict[str, Any],
    material_name: Optional[str] = None,
) -> Optional[str]:
    """
    Extract a countryAlpha2 code from either a single material lookup record or a full lookup payload.

    Args:
        lookup_data (Dict[str, Any]): Either one material lookup dictionary or the full lookup wrapper.
        material_name (Optional[str]): Specific material name to extract from a full lookup payload.

    Steps:
        1. Validate that the lookup input is a dictionary.
        2. Detect whether the input already represents a single material record.
        3. Extract the country code directly when a single material record is provided.
        4. Otherwise inspect the top-level `materials` dictionary.
        5. Use the requested material name when one is supplied.
        6. Fall back to the only material entry when exactly one material exists.
        7. Return the extracted country code or None when no unambiguous value is available.

    Returns:
        Optional[str]: Extracted countryAlpha2 code when available, otherwise None.

    Notes:
        - When a full lookup payload contains multiple materials and `material_name` is omitted,
          no value is returned because the target entry would be ambiguous.
    """
    if not isinstance(lookup_data, dict):
        return None

    if "raw" in lookup_data:
        return _extract_country_from_single_material(lookup_data)

    materials = lookup_data.get("materials")
    if not isinstance(materials, dict) or not materials:
        return None

    if material_name:
        material_dict = materials.get(material_name)
        if not isinstance(material_dict, dict):
            return None
        return _extract_country_from_single_material(material_dict)

    if len(materials) == 1:
        only_material_dict = next(iter(materials.values()))
        if isinstance(only_material_dict, dict):
            return _extract_country_from_single_material(only_material_dict)

    return None
