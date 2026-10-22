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
from pathlib import Path

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
)

from configuration_dialog import ConfigurationDialog
from material_lookup_controller import MaterialLookupController
from cx_material_consumer.config import Settings
from material_update_dialog import (
    open_or_create_material_update_from_import,
    extract_country_alpha2_from_lookup_material,
)


class MaterialImportDialog(QDialog):
    """
    Desktop dialog for looking up supplier material data, showing the returned
    JSON payload, and adding the selected material to the local materials
    database before optionally opening the material update workflow.

    Class-Level Constants:
        None.

    Attributes:
        lookup_controller (MaterialLookupController): Controller used to execute
            material lookups and merge lookup results into the materials database.
        last_result (Any): Stores the most recent successful lookup result so it
            can be added to the database later.
        configuration_dialog (ConfigurationDialog | None): Cached configuration
            dialog instance used to edit real-mode settings.
        supplier_material_number_edit (QLineEdit): Input field for the supplier
            material number.
        supplier_bpnl_edit (QLineEdit): Optional input field for the supplier BPNL.
        site_bpns_edit (QLineEdit): Optional input field for the site BPNS.
        materials_db_path_edit (QLineEdit): Input field for the target materials
            database JSON path.
        mock_mode_checkbox (QCheckBox): Checkbox that controls whether lookup runs
            in mock mode or real Catena-X mode.
        status_label (QLabel): Label used to display short status messages.
        output_text (QPlainTextEdit): Read-only area used to display formatted
            lookup output as JSON text.

    Methods:
        __init__(): Initialize the dialog, controller, and internal state.
        _build_ui(): Create and arrange all widgets used by the dialog.
        _set_status(): Update the visible status message.
        open_configuration_dialog(): Open or reactivate the configuration dialog.
        _ensure_real_mode_is_configured(): Verify that required real-mode
            settings are available before lookup starts.
        on_lookup(): Execute the lookup flow with the current form values and
            display the result.
        on_add_to_db(): Add the most recent lookup result to the materials
            database and open the update dialog for the imported material.
    """

    def __init__(self, parent=None):
        """
        Initialize the material import dialog, create the lookup controller, and
        prepare the dialog state before the user interacts with the UI.

        Args:
            parent (Any): Optional parent widget that owns this dialog and any
                child dialogs or message boxes opened from it.

        Steps:
            1. Initialize the QDialog base class and configure the window title
               and initial size.
            2. Create the MaterialLookupController and connect it to the dialog's
               mock-mode checkbox and status callback.
            3. Initialize state used to store the last lookup result and the
               optional configuration dialog instance.
            4. Build the visible user interface for the dialog.

        Returns:
            None: This initializer configures the dialog instance in place.

        Notes:
            - The lookup controller reads the mock-mode state through a lambda so
              it always uses the current checkbox value.
            - The UI is built after the controller-related state is prepared.
        """
        super().__init__(parent)
        self.setWindowTitle("Import Supplier Material")
        self.resize(900, 700)

        self.lookup_controller = MaterialLookupController(
            mock_mode_getter=lambda: self.mock_mode_checkbox.isChecked(),
            status_callback=self._set_status,
        )

        self.last_result = None
        self.configuration_dialog = None

        self._build_ui()

    def _build_ui(self):
        """
        Create and arrange all widgets used by the material import dialog.

        Args:
            None.

        Steps:
            1. Create the main layout and the form layout for user input fields.
            2. Create line edits for material lookup inputs and set the default
               materials database path to a sibling "materials_validation.json"
               file.
            3. Create the mock-mode checkbox and add all inputs to the form.
            4. Create the status label and read-only output area for lookup
               results.
            5. Create the action buttons, connect them to their handlers, and
               add them to the dialog layout.

        Returns:
            None: This method constructs the dialog widgets and attaches them to
            the current instance.

        Notes:
            - The output text box is read-only because lookup results are
              displayed for review rather than edited directly here.
            - The default database path is resolved relative to the current file.
        """
        layout = QVBoxLayout(self)

        form_layout = QFormLayout()

        self.supplier_material_number_edit = QLineEdit()
        self.supplier_bpnl_edit = QLineEdit()
        self.site_bpns_edit = QLineEdit()
        self.materials_db_path_edit = QLineEdit()

        self.materials_db_path_edit.setText(
            str(Path(__file__).resolve().parent / "app_data/materials_validation.json")
        )

        self.mock_mode_checkbox = QCheckBox("Mock mode")
        self.mock_mode_checkbox.setChecked(True)

        form_layout.addRow("Supplier material number*", self.supplier_material_number_edit)
        form_layout.addRow("Supplier BPNL", self.supplier_bpnl_edit)
        form_layout.addRow("Site BPNS", self.site_bpns_edit)
        form_layout.addRow("Materials DB path", self.materials_db_path_edit)
        form_layout.addRow("", self.mock_mode_checkbox)

        layout.addLayout(form_layout)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text, stretch=1)

        button_row = QHBoxLayout()

        self.lookup_button = QPushButton("Lookup")
        self.lookup_button.clicked.connect(self.on_lookup)

        self.add_to_db_button = QPushButton("Add to materials DB")
        self.add_to_db_button.clicked.connect(self.on_add_to_db)

        self.configuration_button = QPushButton("Configuration")
        self.configuration_button.clicked.connect(self.open_configuration_dialog)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)

        button_row.addWidget(self.lookup_button)
        button_row.addWidget(self.add_to_db_button)
        button_row.addWidget(self.configuration_button)
        button_row.addStretch(1)
        button_row.addWidget(self.close_button)

        layout.addLayout(button_row)

    def _set_status(self, message: str):
        """
        Update the status label with the latest message for the user.

        Args:
            message (str): Text to display in the status label. Empty values are
                normalized to an empty string.

        Steps:
            1. Receive the status message from the dialog or controller.
            2. Normalize empty input to an empty string.
            3. Update the status label text shown in the dialog.

        Returns:
            None: This method updates the UI in place.

        Notes:
            - This method is used as the status callback for the lookup
              controller.
        """
        self.status_label.setText(message or "")

    def open_configuration_dialog(self):
        """
        Open the configuration dialog used to edit real-mode lookup settings.

        Args:
            None.

        Steps:
            1. Create the configuration dialog if it has not been opened before.
            2. Reload the existing values into the form when reusing a cached
               dialog instance.
            3. Show the configuration dialog and bring it to the foreground.

        Returns:
            None: This method opens or reactivates the configuration dialog.

        Notes:
            - The dialog instance is cached so repeated openings reuse the same
              window.
            - Reloading values keeps the configuration form synchronized before
              it is shown again.
        """
        if self.configuration_dialog is None:
            self.configuration_dialog = ConfigurationDialog(parent=self)
        else:
            self.configuration_dialog._load_values_into_form()
        self.configuration_dialog.show()
        self.configuration_dialog.raise_()
        self.configuration_dialog.activateWindow()

    def _ensure_real_mode_is_configured(self) -> bool:
        """
        Check that all required settings for real Catena-X lookup mode are
        present before a lookup is started.

        Args:
            None.

        Steps:
            1. Return immediately when mock mode is enabled because no real-mode
               configuration is required in that case.
            2. Ask the lookup controller which real-mode settings are missing.
            3. Build a user-friendly warning message when required settings are
               not configured.
            4. Display the warning, update the dialog status text, and stop the
               lookup flow.

        Returns:
            bool: True when lookup may continue, or False when required
            real-mode settings are missing.

        Notes:
            - Missing setting names are converted to user-facing labels through
              Settings.label_for_env_key(...).
            - This method prevents avoidable lookup failures by validating
              configuration before the controller runs.
        """
        if self.mock_mode_checkbox.isChecked():
            return True

        missing = self.lookup_controller.missing_real_mode_fields()
        if not missing:
            return True

        bullets = "\n".join(
            f"- {Settings.label_for_env_key(key)}" for key in missing
        )
        message = (
            "Real Catena-X mode is not configured yet.\n\n"
            "Please click on 'Configuration' and fill in at least the required configuration aspects before trying a lookup again.\n\n"
            f"Missing required settings:\n{bullets}"
        )
        QMessageBox.warning(self, "Real mode not configured", message)
        self._set_status(
            "Lookup stopped. Open Configuration and fill in the required real-mode settings."
        )
        return False

    def on_lookup(self):
        """
        Read the current input values, run the material lookup, and display the
        resulting JSON payload in the dialog.

        Args:
            None.

        Steps:
            1. Read and normalize the current values from the supplier and site
               input fields.
            2. Validate that a supplier material number is present.
            3. Verify that real-mode configuration is complete when mock mode is
               disabled.
            4. Execute the lookup through the MaterialLookupController.
            5. Store the result, render the returned output as formatted JSON,
               and update the status message.

        Returns:
            None: This method updates dialog state and UI elements in place.

        Notes:
            - supplier_bpnl and site_bpns are converted to None when left empty.
            - The material_key argument is currently passed as None so the
              controller determines the relevant key from the lookup flow.
            - Exceptions are shown to the user in a critical message box.
        """
        supplier_material_number = self.supplier_material_number_edit.text().strip()
        supplier_bpnl = self.supplier_bpnl_edit.text().strip() or None
        site_bpns = self.site_bpns_edit.text().strip() or None
        material_key = None

        if not supplier_material_number:
            QMessageBox.warning(self, "Missing input", "Please enter a supplier material number.")
            return

        if not self._ensure_real_mode_is_configured():
            return

        try:
            result = self.lookup_controller.run_lookup(
                parent=self,
                supplier_material_number=supplier_material_number,
                supplier_bpnl=supplier_bpnl,
                site_bpns=site_bpns,
                material_key=material_key,
            )
            if not result:
                return

            self.last_result = result

            self.output_text.setPlainText(
                json.dumps(result.output, indent=2, ensure_ascii=False)
            )

            self._set_status(f'Material "{result.material_key}" loaded successfully.')

        except Exception as exc:
            QMessageBox.critical(self, "Lookup failed", str(exc))

    def on_add_to_db(self):
        """
        Add the most recent lookup result to the materials database and then
        open the material update dialog for additional editing.

        Args:
            None.

        Steps:
            1. Confirm that a successful lookup result is available.
            2. Validate that a target materials database path has been provided.
            3. Merge the last lookup result into the database through the lookup
               controller.
            4. Show a success message and update the dialog status text.
            5. Extract the lookup country code, open the update workflow for the
               imported material, and close the dialog when the update succeeds.

        Returns:
            None: This method performs UI updates and database-side actions but
            does not return a value.

        Notes:
            - The follow-up update dialog uses the selected materials database path.
            - The imported material name comes from self.last_result.material_key.
            - Exceptions during merge or update startup are surfaced in a
              critical message box.
        """
        if not self.last_result:
            QMessageBox.information(self, "No material", "Please run a lookup first.")
            return

        database_path = self.materials_db_path_edit.text().strip()
        if not database_path:
            QMessageBox.warning(self, "Missing path", "Please enter the materials DB path.")
            return

        try:
            self.lookup_controller.merge_last_result_into_db(database_path)
            QMessageBox.information(
                self,
                "Success",
                f'Material "{self.last_result.material_key}" was added to the database.',
            )
            self._set_status(f"Merged into {database_path}")

            lookup_material = self.last_result.output
            lookup_country_alpha2 = extract_country_alpha2_from_lookup_material(lookup_material)

            json_path = database_path
            material_name = self.last_result.material_key

            updated = open_or_create_material_update_from_import(
                self,
                json_path,
                material_name,
                lookup_country_alpha2
            )

            if updated:
                self.accept()

        except Exception as exc:
            QMessageBox.critical(self, "Merge failed", str(exc))
