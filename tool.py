""""    
    Copyright (C) 2026  Digiser, Riech & Nebel

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
import sys
import json
import math
import re
import uuid
from typing import Any, Dict, List, Iterable, Optional, Tuple
from collections import defaultdict, Counter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QAction,
    QDialog,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QPushButton,
    QMessageBox,
    QWidget,
    QFileDialog, 
    QTreeWidget, 
    QTreeWidgetItem,
    QScrollArea,
    QGroupBox, 
    QComboBox, 
    QSizePolicy, 
    QDialogButtonBox, 
    QListWidget, 
    QHBoxLayout, 
    QFormLayout,
    QCheckBox,
    QSpinBox, 
    QTableWidget,
    QTableWidgetItem,
    QInputDialog,
    QDoubleSpinBox, 
    QSplitter,
    QStyledItemDelegate,
    QProgressBar,
    QFrame
)
from PyQt5.QtCore import (
    Qt,
    QFile,
    QTextStream,
    QTimer,
)
from PyQt5.QtGui import (
    QFont,
    QIcon
)

import logging
from dataclasses import dataclass

from material_import_dialog import MaterialImportDialog

from pathlib import Path
from material_update_dialog import open_material_update_dialog
from assembly_time_service import compute_joining_assembly_time
from classification_service import (
    collect_part_classifications,
    find_parent_classification,
    update_assembly_classifications,
)
from tool_change_planner import compute_min_tool_changes
from r_strategy_weights import validate_goal_weights
from mirror_assigned_connections import (
    mirror_empty_assigned_connections_and_delete,
    synchronize_assigned_connections_by_richer_side,
)

import ctypes

from ast import literal_eval
from copy import deepcopy

BASE_DIR = Path(__file__).resolve().parent

def apply_dynamic_stylesheet(app, window, base_resolution=(1920, 1080)):
    """
    Apply a dynamic stylesheet to the application based on the screen resolution
    where the main window is currently displayed.

    This function adjusts the application's stylesheet and font size dynamically
    to ensure proper scaling on different screen resolutions. It also optionally
    resizes the main window based on the calculated scale factor.

    Args:
        app (QApplication): The main application instance.
        window (QMainWindow): The main application window.
        base_resolution (tuple): The base resolution (width, height) used as a reference
                                 for scaling. Defaults to (1920, 1080).

    Steps:
        1. Determine the current screen's resolution.
        2. Calculate a scale factor based on the ratio of the current resolution
           to the base resolution.
        3. Adjust the global font size using the scale factor.
        4. Load and scale the application's stylesheet properties (e.g., font-size,
           padding, margin, etc.) using the scale factor.
        5. Apply the scaled stylesheet to the application.
        6. Optionally resize the main window based on the scale factor.

    Notes:
        - The function expects a stylesheet file named "style.qss" to be present
          in the working directory.
        - If the stylesheet file cannot be opened, the function will print an error
          message and exit without applying any changes.

    Example:
        apply_dynamic_stylesheet(app, main_window)
    """
    # Determine the current screen
    screen = window.screen() if hasattr(window, 'screen') else app.primaryScreen()
    screen_size = screen.size()
    width, height = screen_size.width(), screen_size.height()
    
    # Calculate scale factor
    scale_factor = min(width / base_resolution[0], height / base_resolution[1])
    
    # Set global font size
    base_font_size = 10
    scaled_font_size = max(int(base_font_size * scale_factor), 8)
    font = QFont("Arial", scaled_font_size)
    app.setFont(font)
    
    # Load stylesheet
    file = QFile(str(BASE_DIR / "style.qss"))
    if not file.open(QFile.ReadOnly | QFile.Text):
        print("Unable to open stylesheet file.")
        return
    
    text_stream = QTextStream(file)
    stylesheet = text_stream.readAll()
    file.close()
    
    # Function to scale stylesheet properties
    def scale_stylesheet(stylesheet, properties, scale):
        """
        Scale specific properties in the stylesheet by the given scale factor.

        Args:
            stylesheet (str): The original stylesheet content.
            properties (list): A list of CSS properties to scale (e.g., font-size, padding).
            scale (float): The scale factor to apply.

        Returns:
            str: The scaled stylesheet content.
        """
        for prop in properties:
            pattern = fr"({prop}\s*:\s*)(\d+)(px)"
            stylesheet = re.sub(
                pattern,
                lambda m: f"{m.group(1)}{int(float(m.group(2)) * scale)}{m.group(3)}",
                stylesheet
            )
        return stylesheet
    
    properties_to_scale = ["font-size", "padding", "margin", "border-radius", "icon-size", "spacing"]
    scaled_stylesheet = scale_stylesheet(stylesheet, properties_to_scale, scale_factor)
    
    # Apply the scaled stylesheet
    app.setStyleSheet(scaled_stylesheet)
    
    # Optionally, resize the window
    original_size = window.size()
    new_width = int(original_size.width() * scale_factor)
    new_height = int(original_size.height() * scale_factor)
    window.resize(new_width, new_height)
            

def load_json_file(file_path: str):
    """
    Load a JSON file and return the parsed data.

    This function attempts to open and parse a JSON file from the specified file path.
    If the file is not found or an error occurs during parsing, an appropriate error
    message is displayed using a QMessageBox, and the function returns None.

    Args:
        file_path (str): The path to the JSON file to be loaded.

    Returns:
        dict or list: The parsed JSON data if successful.
        None: If the file is not found or an error occurs during parsing.

    Error Handling:
        - Displays a warning message if the file is not found.
        - Displays a warning message if any other exception occurs during file loading.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        QMessageBox.warning(None, "Error", 
                            f"The file at {file_path} could not be found.")
        return None
    except Exception as e:
        QMessageBox.warning(None, "Error", 
                            f"An error occurred while loading the JSON file: {e}")
        return None

def read_file(file_path: str) -> str:
    """
    Read a file and return its contents as a string.

    This function attempts to open and read the contents of a file specified by the
    given file path. If the file is not found or an error occurs during reading,
    an appropriate error message is displayed using a QMessageBox, and the function
    returns None.

    Args:
        file_path (str): The path to the file to be read.

    Returns:
        str: The contents of the file as a string if successful.
        None: If the file is not found or an error occurs during reading.

    Error Handling:
        - Displays a warning message if the file is not found.
        - Displays a warning message if any other exception occurs during file reading.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        QMessageBox.warning(None, "Error", 
                            f"The file at {file_path} could not be found.")
        return None
    except Exception as e:
        QMessageBox.warning(None, "Error", 
                            f"An error occurred while reading the JSON file: {e}")
        return None

def dump_file(data: dict, file_path: str) -> None:
    """
    Dump data to a JSON file.

    This function writes the provided dictionary data to a JSON file at the specified
    file path. If the operation is successful, a success message is displayed using
    a QMessageBox. If an error occurs during the process, an appropriate error message
    is displayed.

    Args:
        data (dict): The data to be written to the JSON file.
        file_path (str): The path to the JSON file where the data will be saved.

    Returns:
        None

    Error Handling:
        - Displays a success message if the data is saved successfully.
        - Displays a warning message if an error occurs during the save operation.
    """
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        QMessageBox.information(None, "Success", 
                                f"Data successfully saved to {file_path}")
    except Exception as e:
        QMessageBox.warning(None, "Error", 
                            f"Failed to save data to JSON: {e}")

class Config:
    """
    A configuration class that holds file paths for various JSON configuration files.

    Attributes:
        category_file (str): Path to the JSON file containing category definitions.
        classification_json (str): Path to the JSON file for classification validation.
        joining_connections_json (str): Path to the JSON file for joining connections.
        materials_json (str): Path to the JSON file for materials validation.
        parameter_to_goal_json (str): Path to the JSON file mapping parameters to goals.
        goal_to_r_strategy_json (str): Path to the JSON file mapping goals to R strategies.
    """
    def __init__(self):
        self.category_file = BASE_DIR / 'categories.json'
        self.classification_json = 'app_data/classification_validation.json'
        self.joining_connections_json = BASE_DIR / 'joining_connections.json'
        self.materials_json = 'app_data/materials_validation.json'
        self.parameter_to_goal_json = 'app_data/parameter_to_goal.json'
        self.goal_to_r_strategy_json = 'app_data/goal_to_r_strategy.json'


class MainWindow(QMainWindow):
    """
    The main application window for the Circular DESIGNer.

    This class represents the primary interface for the application. It manages
    the user interface, menus, and interactions, and provides methods for handling
    various application functionalities such as project management, material handling,
    and parameter assignments.

    Attributes:
        responses (list): Stores user responses during the assessment process.
        current_project_data (dict): Holds the data for the currently loaded project.
        current_project_file (str or None): Path to the currently loaded project file.
        materials_json (str): Path to the materials JSON file from the configuration.

    Methods:
        __init__(config): Initializes the main application window with the given configuration.
        init_main_window_ui(): Sets up the main window's user interface.
        create_menu(): Creates the main menu bar and its actions.
        connect_screen_changed(): Connects the screenChanged signal to dynamically update the stylesheet.
        update_stylesheet(): Updates the application's stylesheet dynamically.
        clear_layout(layout): Removes all widgets from a given layout.
        new_project(): Handles the creation of a new project.
        start_assessment(): Starts the assessment process.
        open_project(): Opens an existing project file.
        save_project(): Saves the current project to a file.
        open_material_catalogue(): Opens the material catalogue for editing.
        open_material_import_dialog(): Opens the material import dialog window and brings it to the front if it already exists.
        def open_update_material_dialog(self): Opens the material update dialog window.
        open_parameter_to_goal_assignment(): Opens the parameter-to-goal assignment editor.
        open_goal_assessment(): Opens the goal assessment editor.
    """

    def __init__(self, config: Config):
        """
        Initialize the main application window.

        This constructor sets up the main application window, initializes the UI, 
        creates the menu, and prepares the necessary attributes for managing the 
        application's state. It also applies a dynamic stylesheet to ensure proper 
        scaling and appearance based on the screen resolution.

        Args:
            config (Config): An instance of the Config class containing file paths 
                             and other configuration settings.

        Returns:
            None
        """
        super().__init__()
        self.init_main_window_ui()
        self.create_menu()
        self.responses = []
        self.current_project_data = {}
        self.current_project_file = None
        self.material_import_dialog = None
        self.materials_json = config.materials_json

        # Apply dynamic stylesheet initially
        app = QApplication.instance()
        if app is not None:
            apply_dynamic_stylesheet(app, self)

    def showEvent(self, event):
        """
        Handle the show event for the main application window.

        This method is triggered when the main window is shown. It ensures that
        the `connect_screen_changed` method is called to handle any screen-related
        changes, such as resolution or scaling adjustments.

        Args:
            event (QShowEvent): The event object associated with the show event.

        Returns:
            None
        """
        super().showEvent(event)
        self.connect_screen_changed()

    def connect_screen_changed(self):
        """
        Connect the screenChanged signal to update the stylesheet dynamically.

        This method ensures that the `update_stylesheet` method is called whenever
        the screen associated with the main window changes (e.g., when the window
        is moved to a different monitor). If the `windowHandle` is not immediately
        available, the method retries after a short delay.

        Returns:
            None
        """
        window_handle = self.windowHandle()
        if window_handle:
            window_handle.screenChanged.connect(self.update_stylesheet)
        else:
            # Retry after a short delay if window_handle is not available
            QTimer.singleShot(100, self.connect_screen_changed)

    def update_stylesheet(self):
        """
        Update the application's stylesheet dynamically.

        This method retrieves the current QApplication instance and applies the
        dynamic stylesheet to the main window. It ensures that the UI scales
        properly based on the current screen resolution or other changes.

        Returns:
            None
        """
        app = QApplication.instance()
        if app:
            apply_dynamic_stylesheet(app, self)

    def init_main_window_ui(self):
        """
        Setup the main window UI elements.

        This method initializes the main window's user interface, including setting
        the window title, creating the central widget, and adding layout and UI
        components such as labels and buttons. It also connects the "Start Analysis"
        button to the `start_assessment` method.

        Returns:
            None
        """
        self.setWindowTitle('Circular DESIGNer')
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        self.welcome_label = QLabel('Welcome to the Circular DESIGNer!', self)
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.welcome_label)

        self.start_button = QPushButton('Start Analysis', self)
        self.start_button.setFixedHeight(50)
        self.start_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.start_button.clicked.connect(self.start_assessment)
        self.main_layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

    def create_menu(self):
        """
        Create the main menu bar.

        This method sets up the main menu bar for the application, including
        various menus and their associated actions. It also adds a project
        information label to the top-right corner of the menu bar.

        Menus and Actions:
            - File:
                - New Project: Triggers the `new_project` method.
                - Open Project: Triggers the `open_project` method.
            - Materials:
                - Add Material: Triggers the `open_material_catalogue` method.
                - Add Material via Catena-X: Triggers the `open_material_import_dialog` method.
                - Update Material: Triggers the `open_update_material_dialog` method.
            - Parameter Assignment:
                - Edit Parameter To Goal Assignments: Triggers the `open_parameter_to_goal_assignment` method.
            - Goals:
                - Edit Goal Weightings: Triggers the `open_goal_assessment` method.

        Attributes:
            project_info_label (QLabel): Displays the current project and version information.

        Returns:
            None
        """
        menu_bar = self.menuBar()
        
        file_menu = menu_bar.addMenu('File')
        file_menu.addAction(QAction('New Project', self, triggered=self.new_project))
        file_menu.addAction(QAction('Open Project', self, triggered=self.open_project))
        
        material_menu = menu_bar.addMenu('Materials')
        material_menu.addAction(QAction('Add Material', self, triggered=self.open_material_catalogue))
        material_menu.addAction(QAction('Add Material via Catena-X', self, triggered=self.open_material_import_dialog))
        material_menu.addAction(QAction('Update Material', self, triggered=self.open_update_material_dialog))
        
        parameter_assignment_menu = menu_bar.addMenu('Parameter Assignment')
        parameter_assignment_menu.addAction(QAction('Edit Parameter To Goal Assignments', self, triggered=self.open_parameter_to_goal_assignment))

        goals_menu = menu_bar.addMenu('Goals')
        goals_menu.addAction(QAction('Edit Goal Weightings', self, triggered=self.open_goal_assessment))

        
        self.project_info_label = QLabel('Project: None | Version: None', self)

        menu_bar.setCornerWidget(self.project_info_label, Qt.TopRightCorner)

    def clear_layout(self, layout):
        """
        Remove all widgets from a layout.

        This method iterates through all items in the given layout, removes them,
        and deletes any associated widgets to free up resources.

        Args:
            layout (QLayout): The layout from which all widgets will be removed.

        Returns:
            None
        """
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def new_project(self):
        """
        Handler for creating a new project.

        This method opens a `ProjectDialog` to gather information for a new project.
        If the dialog is accepted, it initializes the current project data and file path
        based on the user's input and updates the project information displayed in the UI.

        Returns:
            None
        """
        dialog = ProjectDialog()
        if dialog.exec_() == QDialog.Accepted:
            self.current_project_data = dialog.project_data
            self.current_project_file = dialog.output_path
            self.update_project_info(dialog.project_data)

    def open_project(self):
        """
        Handler for opening an existing project.

        This method opens a file dialog to allow the user to select an existing project
        file in JSON format. If a file is selected, it loads the project data, updates
        the current project information displayed in the UI, and sets the current project
        data and file path.

        Steps:
            1. Open a file dialog to select a JSON project file.
            2. Load the selected file using the `load_json_file` function.
            3. Update the project information in the UI.
            4. Store the loaded project data and file path in the corresponding attributes.

        Args:
            None

        Returns:
            None
        """
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Project', '', 
                                                   'JSON Files (*.json)')
        if file_name:
            project_data = load_json_file(file_name)
            self.update_project_info(project_data)
            self.current_project_data = project_data
            self.current_project_file = file_name

    def open_material_catalogue(self):
        """
        Open the MaterialParameterDialog from the MainWindow.

        This method allows the user to add or update material information in the materials database.
        It prompts the user to enter a material name, opens a dialog for entering material parameters,
        and updates the materials JSON file with the provided data.

        Steps:
            1. Prompt the user to enter a material name using an input dialog.
            2. If a valid name is provided, open the MaterialParameterDialog.
            3. Collect the material data entered by the user.
            4. Load the existing materials database from the materials JSON file.
            5. Add or update the material data in the database.
            6. Save the updated database back to the JSON file.
            7. Display a success message if the operation is successful.

        Args:
            None

        Returns:
            None

        Error Handling:
            - Displays a warning message if the material name is empty.
            - Ensures the "materials" key exists in the database before updating it.
        """
        # Prompt the user to enter a material name
        material_name, ok = QInputDialog.getText(self, "Material Name", "Enter the name of the material:")
        
        if ok and material_name.strip():  # Proceed only if the user provides a valid name
            dialog = MaterialParameterDialog(material_name.strip(), self)
            if dialog.exec_() == QDialog.Accepted:
                # Get the data entered by the user
                material_data = dialog.get_data()
                
                # Update the materials database with the collected data
                material_file_path = self.materials_json  # Adjust the path if necessary
                materials_db = load_json_file(material_file_path)
                
                if "materials" not in materials_db:
                    materials_db["materials"] = {}
                
                materials_db["materials"][material_name.strip()] = material_data
                dump_file(materials_db, material_file_path)
                
                QMessageBox.information(self, "Success", f"Material '{material_name}' has been added/updated.")
        else:
            QMessageBox.warning(self, "Invalid Input", "Material name cannot be empty.")

    def open_material_import_dialog(self):
        """
        Opens the material import dialog window and brings it to the front if it already exists.

        Args:
            self: The main window instance that owns the material import dialog.

        Steps:
            1. Check whether the material import dialog has already been created.
            2. Create a new MaterialImportDialog instance with the current window as parent if no dialog exists yet.
            3. Show the dialog window.
            4. Raise the dialog above other windows.
            5. Activate the dialog so it receives keyboard focus.

        Returns:
            None
        """
        if self.material_import_dialog is None:
            self.material_import_dialog = MaterialImportDialog(parent=self)

        self.material_import_dialog.show()
        self.material_import_dialog.raise_()
        self.material_import_dialog.activateWindow()

    def open_update_material_dialog(self):
        """
        Opens the material update dialog using the materials validation JSON file
        located in the same directory as the current Python file.

        Args:
            None.

        Steps:
            1. Open the material update dialog using the current instance and the JSON path.

        Returns:
            None: This method opens the dialog but does not return a value.

        Notes:
            - Calls open_material_update_dialog(self, materials_json) to display the dialog.
        """
        open_material_update_dialog(self, self.materials_json)

    def open_goal_assessment(self): 
        """
        Open the GoalDialog to edit goal weightings.

        This method allows the user to modify goal weightings by opening the `GoalDialog`.
        If the dialog is accepted, the updated goal data is retrieved for further processing.

        Args:
            None

        Returns:
            None

        Steps:
            1. Create an instance of the `GoalDialog` with the current configuration.
            2. Display the dialog to the user.
            3. If the dialog is accepted, retrieve the updated goal data.
        """
        try:
            dialog = GoalDialog(Config())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid goal weights", str(exc))
            return
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()

    def open_parameter_to_goal_assignment(self): 
        """
        Open the ParameterAssignmentDialog to edit parameter-to-goal assignments.

        This method allows the user to modify the mapping of parameters to goals by opening
        the `ParameterAssignmentDialog`. If the dialog is accepted, the updated assignment
        data is retrieved for further processing.

        Args:
            None

        Returns:
            None

        Steps:
            1. Create an instance of the `ParameterAssignmentDialog` with the current configuration.
            2. Display the dialog to the user.
            3. If the dialog is accepted, retrieve the updated parameter-to-goal assignment data.
        """
        dialog = ParameterAssignmentDialog(Config())
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()
    
    
    def update_project_info(self, project_data):
        """
        Update the project info label with the project name and version.

        This method updates the `project_info_label` to display the current project's
        name and version based on the provided project data.

        Args:
            project_data (dict): A dictionary containing project information, including
                                'project_name' and 'version' keys.

        Returns:
            None

        Notes:
            - If 'project_name' or 'version' is not found in the project data, it defaults
            to 'Unknown'.
        """
        project_name = project_data.get('project_name', 'Unknown')
        version = project_data.get('version', 'Unknown')
        self.project_info_label.setText(f'Project: {project_name} | Version: {version}')

    def start_assessment(self):
        """
        Start the assessment process.

        This method initiates the assessment process by first ensuring that a project
        is loaded or created. It then opens a selection dialog for the user to make
        necessary selections. If the selection is accepted, it proceeds to open a
        calculation dialog to perform the assessment calculations.

        Steps:
            1. Check if a project file is loaded or created. If not, display a warning.
            2. Open the `SelectionDialog` for user input.
            3. If the selection is accepted:
                - Reload the updated JSON file (otherwise the self.data is empty).
                - Open the `CalculationDialog` to perform calculations.
                - If the calculation is successful, print a success message.
                - If the calculation is canceled, display a warning.
            4. If the selection is canceled, display an informational message.

        Args:
            None

        Returns:
            None
        """
        if not self.current_project_file:
            QMessageBox.warning(self, 'Error', 'Load or create a project first.')
            return

        # Open the selection dialog
        selection_dialog = SelectionDialog(self.current_project_data, self.current_project_file, config)
        if selection_dialog.exec_() == QDialog.Accepted:
            
            # Open the updated JSON file
            with open(self.current_project_file, "r", encoding="utf-8") as f:
                self.current_project_data = json.load(f)

            # Open the calculation dialog
            calculation_dialog = CalculationDialog(self.current_project_data, self.current_project_file, config)

            if calculation_dialog.exec_() == QDialog.Accepted:
                print("Calculation successful")
            else:
                QMessageBox.warning(self, 'Error', 'Calculation cancelled.')

        else:
            QMessageBox.information(self, 'Information', 'Assessment cancelled.')

    def _hide_start_ui(self):
        """
        Hide the initial UI elements.

        This method hides the "Start Analysis" button and the welcome label
        from the main window. It is typically used to transition from the
        initial UI to the main application interface.

        Args:
            None

        Returns:
            None
        """
        self.start_button.setVisible(False)
        self.welcome_label.setVisible(False)

class EditableColumnsDelegate(QStyledItemDelegate):
    """
    Restrict inline editing to Volume (1), Density (2), and Material (3) columns.
    All other columns return no editor and are therefore read-only.
    """
    _EDITABLE = {1, 2, 3}

    def createEditor(self, parent, option, index):
        return (
            super().createEditor(parent, option, index)
            if index.column() in self._EDITABLE
            else None
        )

class ProjectDialog(QDialog):
    """
    A dialog for managing project-related data and operations.

    This class provides a user interface for handling project data, including
    parsing STEP files and extracting relevant information such as structure,
    material names, volume, and density. It also defines regex patterns for
    extracting specific data from STEP files.

    Class-Level Constants:
        STRUCTURE_PATTERN (str): Regex pattern for extracting product definition shapes.
        NAME_MATERIAL_PATTERN (str): Regex pattern for extracting product names and materials.
        VOLUME_PATTERN (str): Regex pattern for extracting volume measurements.
        DENSITY_PATTERN (str): Regex pattern for extracting density measurements.

    Attributes:
        step_file_path (str): Path to the STEP file being processed.
        components_dict (dict): Dictionary to store component data extracted from the STEP file.
        hierarchy (dict): Dictionary to store the hierarchical structure of components.

    Methods:
        __init__(): Initializes the dialog and its UI components.
        init_ui(): Sets up the user interface for the dialog.
        parse_step_file(): Parses the STEP file and extracts relevant data.
        update_hierarchy(): Updates the hierarchy based on the extracted data.
    """

    try:
        import cadquery as cq
        from OCP.TDF import TDF_Label, TDF_LabelSequence
    except Exception as e:
        print("ERROR: CadQuery/OCP missing.", file=sys.stderr); raise

    # Define regex patterns as class-level constants
    STRUCTURE_PATTERN = (
        r"#\d+\s*=\s*PRODUCT_DEFINITION_SHAPE\s*\(\s*"
        r"'Placement\s+#\d+'\s*,\s*"
        r"'Placement\s+of\s+([\w\-_]+)\s+"
        r"with\s+respect\s+to\s+([\w\-_]+)'\s*,\s*#\d+\s*\)\s*;"
    )    

    NAME_MATERIAL_PATTERN = (
        r"#\d+\s*=\s*PRODUCT\s*\(\s*'([^']+)'\s*,\s*"
        r"'([^']+)'\s*,\s*'NOT\s+SPECIFIED'\s*,\s*\(\s*#\d+\s*\)\s*\)\s*;"
    )

    VOLUME_PATTERN = (
        r"#\d+\s*=\s*MEASURE_REPRESENTATION_ITEM\s*\(\s*'volume\s+measure'\s*,\s*"
        r"VOLUME_MEASURE\s*\(\s*([\d]+(?:\.\d+)?[Ee][+-]?\d+)\s*\)\s*,\s*#\d+\s*\)\s*;"
    )

    DENSITY_PATTERN = (
        r"#\d+\s*=\s*MEASURE_REPRESENTATION_ITEM\s*\(\s*'density\s+measure'\s*,\s*"
        r"POSITIVE_RATIO_MEASURE\s*\(\s*([\d]+(?:\.\d+)?[Ee][+-]?\d+)\s*\)\s*,\s*#\d+\s*\)\s*;"
    )

    QUOTED = r"'(?:[^']|'')*'"
    ATTR   = rf"(?:{QUOTED}|\$)"

    NAMES_PATTERN = (
        r"(#\d+)\s*=\s*PRODUCT\s*\(\s*"
        rf"(?P<a1>{ATTR})\s*,\s*(?:{ATTR})\s*,\s*(?:{ATTR})\s*,\s*\([^)]*\)\s*\)\s*;?"
    )

    FORMATION_TO_PRODUCT_PATTERN = (
        r"(#\d+)\s*=\s*PRODUCT_DEFINITION_FORMATION(?:_WITH_SPECIFIED_SOURCE)?\s*\(\s*"
        rf"{ATTR}\s*,\s*{ATTR}\s*,\s*(?P<pdef>#\d+)\s*[^)]*\)"
    )

    DEFINITION_TO_FORMATION_PATTERN = (
        r"(#\d+)\s*=\s*PRODUCT_DEFINITION\s*\(\s*"
        rf"{ATTR}\s*,\s*{ATTR}\s*,\s*(#\d+)\s*[^)]*\)"
    )

    USAGE_PATTERN = (
        r"#\d+\s*=\s*NEXT_ASSEMBLY_USAGE_OCCURRENCE\s*\(\s*"
        rf"{ATTR}\s*,\s*{ATTR}\s*,\s*{ATTR}\s*,\s*"
        r"(?P<relating>#\d+)\s*,\s*(?P<related>#\d+)\s*,\s*[^)]*\s*\)"
    )

    LOG = logging.getLogger("step_volume_report")
    UNIT_TO_MM = {"mm":1.0,"cm":10.0,"m":1000.0,"in":25.4,"ft":304.8}
    DEFAULT_DENSITY_KG_PER_MM3 = None

    def __init__(self):
        """
        Initialize the ProjectDialog.

        This constructor sets up the dialog's user interface and initializes
        attributes for managing project data.

        Args:
            None

        Returns:
            None
        """
        super().__init__()
        self.init_ui()
        self.step_file_path = ""
        self.components_dict = {}
        self.hierarchy = {}
        
    def init_ui(self):
        """
        Initialize the user interface for the dialog.

        This method sets up the UI components for the dialog, including input fields,
        buttons, and a tree widget for displaying and managing STEP file data. It also
        connects the buttons to their respective event handlers.

        UI Components:
            - Project Name Input: A QLineEdit for entering the project name.
            - Version Input: A QLineEdit for entering the project version.
            - File Selection Label: A QLabel to display the selected file name.
            - Load Button: A QPushButton to load a STEP file.
            - Process Button: A QPushButton to process the loaded STEP file (initially disabled).
            - Tree Widget: A QTreeWidget to display component data with columns for name, volume,
            density, material, mass, and total mass.
            - Refresh Tree Button: A QPushButton to refresh the tree widget.
            - Delete Button: A QPushButton to delete the selected component from the tree.
            - Add Component Button: A QPushButton to add a new component to the tree.
            - Save Changes Button: A QPushButton to save all changes to the project.

        Args:
            None

        Returns:
            None
        """
        self.setWindowTitle("Load STEP File")
        self.setGeometry(100, 100, 700, 500)
        self.layout = QVBoxLayout(self)

        # Project name input
        self.project_name_input = QLineEdit(self)
        self.project_name_input.setPlaceholderText("Enter Project Name")
        self.layout.addWidget(self.project_name_input)
        self.project_name = self.project_name_input.text()

        # Version input
        self.version_input = QLineEdit(self)
        self.version_input.setPlaceholderText("Enter Version")
        self.layout.addWidget(self.version_input)
        self.version = self.version_input.text()


        # File selection label
        self.file_label = QLabel("No file selected", self)
        self.layout.addWidget(self.file_label)

        # Load button
        self.load_button = QPushButton("Load STEP File", self)
        self.load_button.clicked.connect(self.load_data_file)
        self.layout.addWidget(self.load_button)

        # Process button
        self.process_button = QPushButton("Process STEP File", self)
        self.process_button.clicked.connect(self.process_step_file)
        self.process_button.setEnabled(False)  # Initially disabled
        self.layout.addWidget(self.process_button)

        # Set up QTreeWidget
        self.tree = QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(["Name", "Volume [mm^3]", "Density [kg/mm^3]", "Material", "Mass [kg]", "Total Mass [kg]"])
        self.layout.addWidget(self.tree)

        # Delete button
        self.delete_button = QPushButton("Delete Selected Component", self)
        self.delete_button.clicked.connect(self.delete_component)
        self.layout.addWidget(self.delete_button)

        # Edit button 
        self.edit_button = QPushButton("Edit Selected Component", self)
        self.edit_button.clicked.connect(self.edit_component)
        self.layout.addWidget(self.edit_button)


        # Add component button
        self.add_button = QPushButton("Add New Component", self)
        self.add_button.clicked.connect(self.add_component)
        self.layout.addWidget(self.add_button)

        # Save changes button
        self.save_button = QPushButton("Save All", self)
        self.save_button.clicked.connect(self.save_all)
        self.layout.addWidget(self.save_button)
    
    def make_duplicate_names_unique(self, input_value):
        """
        Creates unique component names by adding numeric suffixes to duplicate names. 
        
        Args: 
            input_value (str | list | dict): A component structure provided as 
                Python-literal text, JSON text, or an already parsed Python list 
                or dictionary. 
        
        Steps: 
            1. Checks whether the input is already a Python list or dictionary. 
            2. Parses text input as Python-literal text and, if necessary, JSON text. 
            3. Creates a deep copy of the component structure. 
            4. Traverses all component dictionaries and their nested components. 
            5. Counts all component names in the structure. 
            6. Adds numeric suffixes to names that occur more than once. 
            7. Returns the updated Python structure without changing the original input. 
            
        Returns: 
            list | dict: A copied component structure where duplicate names are 
            made unique. For example, repeated names such as ``PLAIN_WASHER_M8`` 
            become ``PLAIN_WASHER_M8_1``, ``PLAIN_WASHER_M8_2``, and so on. 
            
        Notes: 
            - Names that occur only once remain unchanged. 
            - Numeric suffixes are assigned in depth-first traversal order. 
            - Existing unique names are protected from collisions with generated suffixes. 
            - The method expects each component dictionary to use a ``components`` key containing a list of nested component dictionaries. 
            - The returned value is a Python list or dictionary, not formatted text. 
            - The method does not modify the original input structure.
        """

        def parse_input(value):
            # The caller already supplied a Python structure.
            if isinstance(value, (list, dict)):
                return value

            if not isinstance(value, str):
                raise TypeError(
                    "Expected Python-literal text, JSON text, list, or dict; "
                    f"received {type(value).__name__}."
                )

            text = value.strip()

            if not text:
                raise ValueError("Input text is empty.")

            # First: Python-literal syntax, e.g. None and single quotes.
            try:
                return literal_eval(text)

            except (SyntaxError, ValueError) as python_error:
                # Second: JSON syntax, e.g. null and double quotes.
                try:
                    return json.loads(text)

                except json.JSONDecodeError as json_error:
                    preview = repr(text[:300])
                    raise ValueError(
                        "Input is neither valid Python-literal text nor valid JSON. "
                        f"Input begins with: {preview}\n"
                        f"Python parse error: {python_error}\n"
                        f"JSON parse error: {json_error}"
                    ) from json_error

        def walk_components(value):
            if isinstance(value, list):
                for item in value:
                    yield from walk_components(item)

            elif isinstance(value, dict):
                yield value

                children = value.get("components", [])
                if not isinstance(children, list):
                    raise TypeError(
                        "'components' must be a list for every component."
                    )

                yield from walk_components(children)

        structure = parse_input(input_value)
        result = deepcopy(structure)

        components = list(walk_components(result))

        names = [
            component["name"]
            for component in components
            if isinstance(component.get("name"), str)
        ]
        name_counts = Counter(names)

        # Prevent generated names from replacing an existing unique name.
        used_names = {
            name
            for name, count in name_counts.items()
            if count == 1
        }
        suffixes = defaultdict(int)

        for component in components:
            name = component.get("name")

            if not isinstance(name, str) or name_counts[name] == 1:
                continue

            suffixes[name] += 1
            new_name = f"{name}_{suffixes[name]}"

            while new_name in used_names:
                suffixes[name] += 1
                new_name = f"{name}_{suffixes[name]}"

            component["name"] = new_name
            used_names.add(new_name)

        return result

    def populate(self):
        """
        Populate the tree widget with component data.

        This method clears the existing tree widget and populates it with data from
        the `complete_structure` attribute. Each root node and its child components
        are added to the tree with relevant details such as name, volume, density,
        material, mass, and total mass. Specific columns are made editable for user
        modifications.

        Steps:
            1. Clear the existing tree widget.
            2. Iterate through the root nodes in `complete_structure`.
            3. Create a `QTreeWidgetItem` for each root node with its details.
            4. Set specific columns (volume and density) as editable.
            5. Add the root node to the tree widget.
            6. Recursively populate child components using `populate_tree`.

        Args:
            None

        Returns:
            None
        """

        self.tree.clear()
        self.complete_structure = self.make_duplicate_names_unique(self.complete_structure)
        for root_node in self.complete_structure:
            root_item = QTreeWidgetItem([
                root_node['name'],
                f"{round(root_node.get('volume [mm^3]', 0), 3)}" if root_node.get('volume [mm^3]') is not None else '',
                f"{root_node.get('density [kg/mm^3]', 0)}" if root_node.get('density [kg/mm^3]') is not None else '',
                f"{root_node.get('material')}" if root_node.get('material') is not None else '',
                f"{round(root_node.get('mass [kg]', 0), 3)}" if root_node.get('mass [kg]') is not None else '',
                f"{round(root_node.get('total mass [kg]', 0), 3)}" if root_node.get('total mass [kg]') is not None else ''
            ])
            root_item.setData(0, Qt.UserRole, root_node['id'])
            # read-only — all editing via dialog
            root_item.setFlags(root_item.flags() & ~Qt.ItemIsEditable)
            self.tree.addTopLevelItem(root_item)
            for child in root_node.get('components', []):
                self.populate_tree(root_item, child)
    
    def update_tree(self):
        """
        Update the tree widget with user-modified data.

        This method collects updated data from the tree widget, validates it, and updates
        the `complete_structure` attribute with the new values. It also identifies any
        missing data fields and displays appropriate warnings. After updating, the tree
        is repopulated and expanded to reflect the changes.

        Steps:
            1. Traverse the tree widget to collect updated data for each node.
            2. Validate the data for required fields (volume, density, material).
            3. Update the `complete_structure` with the collected data.
            4. Repopulate the tree widget with the updated structure.
            5. Display a success message if the update is successful.
            6. If any data is missing, display a warning with details of the missing fields.

        Args:
            None

        Returns:
            None

        Error Handling:
            - Displays a critical error message if an exception occurs during the update process.
            - Displays a warning message if any required data fields are missing.

        Notes:
            - The `complete_structure` attribute is recalculated using the `build_structure` method.
            - The tree is expanded after repopulation to show all nodes.

        Example:
            self.update_tree()
        """
        try:
            # Collect updated data from the tree
            updated_node_map = {}
            self.missing_data = defaultdict(list)
        
            def traverse(item):
                node_id = item.data(0, Qt.UserRole)
                if node_id and not item.childCount():
                    # Leaf node: read data
                    volume = float(item.text(1)) if item.text(1) else None
                    density = float(item.text(2)) if item.text(2) else None
                    material = item.text(3) if item.text(3) else None
                
                    if volume is None:
                        self.missing_data[node_id].append('Volume')
                    if density is None:
                        self.missing_data[node_id].append('Density')
                    if material is None:
                        self.missing_data[node_id].append('Material')
                
                    updated_node_map[node_id] = {
                        'volume [mm^3]': volume,
                        'density [kg/mm^3]': density,
                        'material': material
                    }
                for i in range(item.childCount()):
                    traverse(item.child(i))
        
            for i in range(self.tree.topLevelItemCount()):
                traverse(self.tree.topLevelItem(i))
        
            # Recalculate the structure with the updated data
            self.complete_structure = self.build_structure(self.complete_structure, updated_node_map)
            
            # Re-populate the tree
            self.populate()
            self.tree.expandAll()
        
            QMessageBox.information(self, "Success", "The structure was successfully updated.")
        
            # Show warning if there is missing data
            if self.missing_data:
                missing_info = ""
                for node_id, fields in self.missing_data.items():
                    node_name = self.node_map.get(node_id, {}).get('name', 'Unknown')
                    missing_info += f"Name: {node_name} - Missing fields: {', '.join(fields)}\n"
            
                QMessageBox.warning(
                    self, 
                    "Missing Data",
                    f"The following data is missing and should be added:\n\n{missing_info}"
                )
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error updating: {e}")
    
    def delete_component(self):
        """
        Delete the selected component from the tree and data structure.

        This method allows the user to delete a selected component from the tree widget
        and the underlying `complete_structure` data. It validates the selection, confirms
        the deletion with the user, and updates the tree and data structure accordingly.

        Steps:
            1. Check if a component is selected in the tree widget.
            2. Validate the selected component's node ID.
            3. Prompt the user for confirmation before deletion.
            4. If confirmed, remove the component from the `complete_structure`.
            5. Re-populate the tree widget to reflect the changes.
            6. Save the updated hierarchy to persist the changes.

        Args:
            None

        Returns:
            None

        Error Handling:
            - Displays a warning if no component is selected or if the selection is invalid.
            - Displays a critical error message if an exception occurs during the deletion process.

        Notes:
            - The `remove_node` method is used to remove the component from the data structure.
            - The tree is expanded after repopulation to show all nodes.

        Example:
            self.delete_component()
        """
        selected_items = self.tree.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a component to delete.")
            return
        
        item = selected_items[0]
        node_id = item.data(0, Qt.UserRole)
        if not node_id:
            QMessageBox.warning(self, "Warning", "Invalid selection.")
            return
        
        reply = QMessageBox.question(
            self, 'Confirm',
            f"Do you really want to delete the component '{item.text(0)}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                # Remove the component from the data structure
                self.complete_structure = self.remove_node(self.complete_structure, node_id)
                
                # Re-populate the tree
                self.populate()
                self.tree.expandAll()
                self.save_hierarchy()
                QMessageBox.information(self, "Success", "The component was deleted.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error deleting: {e}")
    
    def remove_node(self, structure, node_id):
        """
        Remove a node from the data structure.

        This method removes a node with the specified `node_id` from the given
        `structure`. The structure can be either a list or a dictionary. The method
        traverses the structure recursively to locate and remove the node.

        Args:
            structure (list or dict): The data structure containing the nodes.
            node_id (str): The unique identifier of the node to be removed.

        Returns:
            list or dict: The updated structure with the specified node removed.

        Notes:
            - If the structure is a list, the method iterates through the list and
            removes the node if found.
            - If the structure is a dictionary, the method iterates through the
            dictionary values and removes the node if found.
            - If the structure is neither a list nor a dictionary, it is returned as-is.
        """
        # Handle case where structure is a list
        if isinstance(structure, list):
            for item in structure:
                if self._remove_recursive(item, node_id):
                    return structure
            return structure

        # Handle case where structure is a dictionary
        if isinstance(structure, dict):
            for root, node in structure.items():
                if self._remove_recursive(node, node_id):
                    return structure
            return structure

        # If structure is neither a list nor a dictionary, return it as-is
        return structure
    
    def _remove_recursive(self, node, node_id):
        """
        Recursively remove a node from the data structure.

        This helper method traverses the `components` of a given node to locate and
        remove a child node with the specified `node_id`. If the node is found and
        removed, the method returns `True`. Otherwise, it continues traversing the
        structure recursively.

        Args:
            node (dict): The current node being traversed.
            node_id (str): The unique identifier of the node to be removed.

        Returns:
            bool: `True` if the node was found and removed, `False` otherwise.

        Notes:
            - The method assumes that the `components` key in the node contains a list
            of child nodes.
            - If the node is removed, the traversal stops, and the method returns `True`.
        """
        if 'components' in node:
            for i, child in enumerate(node['components']):
                if child['id'] == node_id:
                    del node['components'][i]
                    return True
                if self._remove_recursive(child, node_id):
                    return True
        return False
    
    def add_node(self, structure, parent_id, new_component):
        """
        Add a new node to the data structure.

        This method adds a new component as a child node to the specified parent node
        in the given `structure`. The structure can be either a list or a dictionary.
        The method traverses the structure recursively to locate the parent node and
        add the new component.

        Args:
            structure (list or dict): The data structure containing the nodes.
            parent_id (str): The unique identifier of the parent node to which the new
                            component will be added.
            new_component (dict): The new component to be added as a child node.

        Returns:
            list or dict: The updated structure with the new component added.

        Notes:
            - If the structure is a list, the method iterates through the list and
            adds the new component to the specified parent node if found.
            - If the structure is a dictionary, the method iterates through the
            dictionary values and adds the new component to the specified parent
            node if found.
            - If the structure is neither a list nor a dictionary, it is returned as-is.
        """
        # Handle case where structure is a list
        if isinstance(structure, list):
            for item in structure:
                if self._add_recursive(item, parent_id, new_component):
                    return structure
            return structure

        # Handle case where structure is a dictionary
        if isinstance(structure, dict):
            for root, node in structure.items():
                if self._add_recursive(node, parent_id, new_component):
                    return structure
            return structure

        # If structure is neither a list nor a dictionary, return it as-is
        return structure

    def _add_recursive(self, node, parent_id, new_component):
        """
        Recursively add a new node to the data structure.

        This helper method traverses the `components` of a given node to locate the
        parent node with the specified `parent_id` and adds the `new_component` as
        a child node. If the parent node is found and the new component is added,
        the method returns `True`. Otherwise, it continues traversing the structure
        recursively.

        Args:
            node (dict): The current node being traversed.
            parent_id (str): The unique identifier of the parent node to which the
                            new component will be added.
            new_component (dict): The new component to be added as a child node.

        Returns:
            bool: `True` if the new component was successfully added, `False` otherwise.

        Notes:
            - The method assumes that the `components` key in the node contains a list
            of child nodes.
            - If the parent node does not have a `components` key, it is created before
            adding the new component.
        """
        # Check if the current node matches the parent_id
        if node.get('name') == parent_id:
            if 'components' not in node:
                node['components'] = []  # Ensure the 'components' key exists
            node['components'].append(new_component)  # Add the new component
            return True

        # Recursively check the 'components' key if it exists
        if 'components' in node:
            for child in node['components']:
                if self._add_recursive(child, parent_id, new_component):
                    return True

        return False
    
    def add_component(self):
        """
        Add a new component to the data structure and UI.

        This method allows the user to add a new component to the `complete_structure`
        data structure. It validates whether a STEP file has been processed, opens a
        dialog to collect the new component's data, and adds the component to the
        selected assembly. The tree widget is updated to reflect the changes.

        Steps:
            1. Check if the `complete_structure` is initialized. If not, display an error.
            2. Retrieve a list of assemblies (nodes with the `components` key).
            3. Open the `AddComponentDialog` to collect data for the new component.
            4. If the dialog is accepted:
                - Retrieve the new component data.
                - Add the new component to the selected assembly in the `complete_structure`.
                - Update the tree widget and expand all nodes.
                - Save the updated hierarchy and display a success message.
            5. If the dialog is canceled, display an informational message.

        Args:
            None

        Returns:
            None

        Error Handling:
            - Displays a critical error message if no STEP file has been processed.
            - Displays an informational message if the operation is canceled.
        """
        # Check if self.complete_structure is None
        if not hasattr(self, 'complete_structure') or self.complete_structure is None:
            QMessageBox.critical(self, "Error", "No STEP file has been processed yet. Please process a STEP file before adding components.")
            return
        else: 
         # Get a list of all assemblies (nodes with 'components' key)
            assemblies = self.get_assemblies(self.complete_structure)
        
        # Open the dialog to get new component data
        dialog = AddComponentDialog(self, assemblies)
        if dialog.exec_() == QDialog.Accepted:
            new_component = dialog.get_data()
            print(new_component)

            # Find the selected assembly and add the new component
            parent_id = new_component.pop('assigned_assembly')
            self.complete_structure = self.add_node(self.complete_structure, parent_id, new_component)

            # Update the UI
            self.populate()
            self.tree.expandAll()
            self.save_hierarchy()
            QMessageBox.information(self, "Success", "The new component was added successfully.")
        else:
            QMessageBox.information(self, "Cancelled", "The operation was cancelled.")

    def get_assemblies(self, structure):
        """
        Recursively collect all assemblies (nodes with a non-empty 'components' key).

        This method traverses the given data structure to identify and collect all
        assemblies. An assembly is defined as a node that contains a non-empty
        `components` key. The method supports both list and dictionary structures.

        Args:
            structure (list or dict): The data structure containing the nodes.

        Returns:
            list: A list of assembly names (nodes with non-empty `components`).

        Notes:
            - If the structure is a list, the method iterates through the list and
            recursively collects assemblies.
            - If the structure is a dictionary, it checks for the `components` key
            and collects the assembly name if the key is non-empty.
        """
        assemblies = []

        if isinstance(structure, list):
            for item in structure:
                assemblies.extend(self.get_assemblies(item))
        elif isinstance(structure, dict):
            # Check if 'components' exists and is not empty
            if 'components' in structure and structure['components']:
                assemblies.append(structure['name'])  # Collect the assembly name
                assemblies.extend(self.get_assemblies(structure['components']))
        return assemblies
    
    def populate_tree(self, parent_item, node):
        """
        Recursively populate the tree widget with child components.

        This method adds a node and its child components to the tree widget. It creates
        a `QTreeWidgetItem` for the given node, sets its data and flags, and recursively
        processes its child components.

        Args:
            parent_item (QTreeWidgetItem): The parent item in the tree to which the current
                                        node will be added as a child.
            node (dict): The data for the current node, including its name, volume, density,
                        material, mass, total mass, and child components.

        Returns:
            None

        Notes:
            - The method sets specific columns (volume and density) as editable for leaf nodes.
            - For non-leaf nodes (nodes with child components), the columns are set as read-only.
            - The `components` key in the node is used to determine and process child nodes.
        """
        item = QTreeWidgetItem([
            node['name'],
            f"{round(node.get('volume [mm^3]'),3)}" if node.get('volume [mm^3]') is not None else '',
            f"{node.get('density [kg/mm^3]')}" if node.get('density [kg/mm^3]') is not None else '',
            f"{node.get('material')}" if node.get('material') is not None else '',
            f"{round(node.get('mass [kg]'),3)}" if node.get('mass [kg]') is not None else '',
            f"{round(node.get('total mass [kg]'),3)}" if node.get('total mass [kg]') is not None else ''
        ])
        parent_item.addChild(item)
    
        item.setData(0, Qt.UserRole, node['id'])
        # read-only — all editing via dialog
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        parent_item.addChild(item)
        for child in node.get('components', []):
            self.populate_tree(item, child)

    def load_data_file(self):
        """
        Prompt the user to select a STEP file and load it.

        This method opens a file dialog to allow the user to select a STEP file. If a file
        is selected, it updates the file label to display the selected file name, stores
        the file path, and enables the process button for further actions.

        Args:
            None

        Returns:
            None

        Notes:
            - The file dialog filters for STEP files (*.step, *.stp) but allows all file types.
            - If no file is selected, the method does nothing.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Select STEP File", 
            "", 
            "STEP Files (*.step *.stp);;All Files (*)", 
            options=options
        )
        
        if file_name:
            self.file_label.setText(file_name)  # Display the selected file name
            self.step_file_path = file_name  # Store the file path
            self.process_button.setEnabled(True)  # Enable the process button

    def ensure_unit(self, u:str)->str:
        u=u.lower(); 
        if u not in self.UNIT_TO_MM: raise ValueError(f"Unknown unit '{u}'")
        return u
    def vol_convert(self, v:float,b:str,o:str)->float:
        return v*(self.UNIT_TO_MM[self.ensure_unit(b)]/self.UNIT_TO_MM[self.ensure_unit(o)])**3
    def area_convert(self, a:float,b:str,o:str)->float:
        return a*(self.UNIT_TO_MM[self.ensure_unit(b)]/self.UNIT_TO_MM[self.ensure_unit(o)])**2

    @dataclass
    class VolumeRow:
        path:str; name:str; kind:str; geom:str; instances:int
        volume_base:float; volume_out:float; volume_mm3:float
        area_base:float; area_out:float; area_mm2:float

    class XDE:
        """
        High-level helper around OCCT XDE (XCAF) for reading STEP files and
        querying assemblies, parts, and shapes.

        Steps:
            1. Create an OCAF/XCAF document to hold the STEP model.
            2. Use STEPCAFControl_Reader to read and transfer the STEP data
            into the document.
            3. Create a ShapeTool handle used to query shape labels and
            assembly structure.

        Args:
            step_path (str): Path to the STEP file to load.
            read_colors (bool): If True, also transfer and store color
                information from the STEP file.
            prefer_definition_name (bool): If True, name resolution prefers
                base/definition (product) labels over instance labels.

        Returns:
            XDE: An initialized helper object wrapping the loaded STEP/XDE
            document and its shape tool.

        Notes:
            Uses STEPCAFControl_Reader, TDocStd_Document, TCollection_ExtendedString,
            XCAFDoc_DocumentTool, and TDataStd_Name.
        """

        try:
            import cadquery as cq
            from OCP.STEPCAFControl import STEPCAFControl_Reader
            from OCP.TDocStd import TDocStd_Document
            from OCP.TCollection import TCollection_ExtendedString
            from OCP.XCAFDoc import XCAFDoc_DocumentTool
            from OCP.TDF import TDF_Label, TDF_LabelSequence
            from OCP.TDataStd import TDataStd_Name
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE
            from OCP.BRepGProp import BRepGProp
            from OCP.GProp import GProp_GProps
        except Exception as e:
            print("ERROR: CadQuery/OCP missing.", file=sys.stderr); raise

        def __init__(self, step_path:str, read_colors:bool=False, prefer_definition_name:bool=False):
            """
            Initialize the XDE helper by loading a STEP file into an XCAF document.

            Steps:
                1. Create a new TDocStd_Document to host the XCAF data.
                2. Configure a STEPCAFControl_Reader to transfer names (and optionally colors).
                3. Read the STEP file and transfer its content into the document.
                4. Obtain an XCAFDoc_ShapeTool handle from the document's main label.
                5. Store the name preference flag for later name resolution.

            Args:
                step_path (str): Path to the STEP file to load.
                read_colors (bool): If True, enable color transfer in the reader.
                prefer_definition_name (bool): If True, later name queries
                    will prefer definition/base labels over instance labels.

            Returns:
                None

            Notes:
                Uses STEPCAFControl_Reader, TDocStd_Document, TCollection_ExtendedString,
                XCAFDoc_DocumentTool.ShapeTool_s.
            """
            self.doc=self.TDocStd_Document(self.TCollection_ExtendedString("XCAF"))
            self.reader=self.STEPCAFControl_Reader(); self.reader.SetNameMode(True)
            if read_colors: self.reader.SetColorMode(True)
            self.reader.ReadFile(step_path); self.reader.Transfer(self.doc)
            self.prefer_definition_name = prefer_definition_name
            self.shape_tool=self.XCAFDoc_DocumentTool.ShapeTool_s(self.doc.Main())
            if self.shape_tool is None: raise RuntimeError("no shape tool")
        def get_free_shapes(self)->Iterable[ProjectDialog.TDF_Label]:
            """
            Yield all free shape labels from the XCAF document.

            Steps:
                1. Create a TDF_LabelSequence container.
                2. Ask the shape tool to fill it with free shape labels.
                3. Yield each label in the sequence one by one.

            Args:
                None

            Returns:
                Iterable[TDF_Label]: An iterator over free shape labels
                representing top-level shapes/assemblies.

            Notes:
                Uses XCAFDoc_ShapeTool.GetFreeShapes to access the root assembly labels.
            """
            seq=self.TDF_LabelSequence(); self.shape_tool.GetFreeShapes(seq)
            for i in range(1,seq.Length()+1): yield seq.Value(i)
        def is_reference(self,label:ProjectDialog.TDF_Label)->bool:
            """
            Check whether a label is a reference (occurrence) in the XCAF structure.

            Steps:
                1. Call ShapeTool.IsReference_s(label).
                2. Convert the result to bool, catching any exceptions.
                3. Return False if the API is not available or fails.

            Args:
                label (TDF_Label): The label to test.

            Returns:
                bool: True if the label is a reference/occurrence, False otherwise.

            Notes:
                Uses XCAFDoc_ShapeTool.IsReference_s.
            """

            try: return bool(self.shape_tool.IsReference_s(label))
            except: return False
        def deref(self,label:ProjectDialog.TDF_Label)->ProjectDialog.TDF_Label:
            """
            Dereference an occurrence label to its underlying definition label.

            Steps:
                1. Check if the label is a reference via IsReference_s.
                2. If it is, call GetReferredShape_s to obtain the referred label.
                3. Return the referred label when available; otherwise return the original label.

            Args:
                label (TDF_Label): The label to dereference.

            Returns:
                TDF_Label: The definition/base label if the input is a reference,
                otherwise the input label itself.

            Notes:
                Uses XCAFDoc_ShapeTool.IsReference_s and GetReferredShape_s.
            """
            try:
                if bool(self.shape_tool.IsReference_s(label)):
                    out=ProjectDialog.TDF_Label(); ok=bool(self.shape_tool.GetReferredShape_s(label,out))
                    if ok: return out
            except: pass
            return label
        def _name_from(self,lab:ProjectDialog.TDF_Label)->Optional[str]:
            """
            Try to read a name string from a given label via TDataStd_Name.

            Steps:
                1. Allocate a TDataStd_Name attribute container.
                2. Use lab.FindAttribute to see if TDataStd_Name is present.
                3. If present, get the underlying string object.
                4. Try multiple string accessors (e.g. ToExtString, ToUTF8CString).
                5. Return the decoded name as a Python string, or None if unavailable.

            Args:
                lab (TDF_Label): The label from which to extract a name.

            Returns:
                Optional[str]: The name string if found; otherwise None.

            Notes:
                Uses TDataStd_Name.GetID_s and TDataStd_Name.Get().
            """
            n=self.TDataStd_Name()
            try:
                if lab.FindAttribute(self.TDataStd_Name.GetID_s(),n):
                    ext=n.Get()
                    for m in ("ToExtString","ToUTF8CString"):
                        try: return getattr(ext,m)()
                        except: pass
                    return str(ext)
            except: pass
            return None
        def get_name(self,label:ProjectDialog.TDF_Label)->str:
            """
            Resolve a human-readable name for a label, honoring the definition/instance preference.

            Steps:
                1. If prefer_definition_name is True:
                    a. Dereference the label to its base/definition label.
                    b. Try to read a name from the base label.
                    c. If that fails, try the original label.
                2. If prefer_definition_name is False:
                    a. First try the label's own name.
                    b. Fall back to the dereferenced base label.
                3. Return "<unnamed>" if no name attribute is found.

            Args:
                label (TDF_Label): The label whose name should be resolved.

            Returns:
                str: The resolved name, or "<unnamed>" if none is available.

            Notes:
                Uses self.deref() and self._name_from().
            """
            if self.prefer_definition_name:
                base=self.deref(label); nm=self._name_from(base)
                if nm: return nm
                nm=self._name_from(label)
                return nm if nm else "<unnamed>"
            else:
                nm=self._name_from(label)
                if nm: return nm
                nm=self._name_from(self.deref(label))
                return nm if nm else "<unnamed>"
        def get_components(self,label:ProjectDialog.TDF_Label)->Iterable[ProjectDialog.TDF_Label]:
            """
            Yield component labels (children) of an assembly or part.

            Steps:
                1. Try instance-first: call ShapeTool.GetComponents_s(label, seq)
                and yield all components if any exist.
                2. If no instance components are found, dereference the label to
                its base and call GetComponents_s(base, seq2).
                3. Yield each component label from the base list.

            Args:
                label (TDF_Label): The label whose components should be iterated.

            Returns:
                Iterable[TDF_Label]: An iterator of component labels.

            Notes:
                Uses self.deref() and XCAFDoc_ShapeTool.GetComponents_s.
            """
            # instance-first
            seq=self.TDF_LabelSequence()
            try:
                self.shape_tool.GetComponents_s(label, seq)
                if seq.Length()>0:
                    for i in range(1,seq.Length()+1): yield seq.Value(i)
                    return
            except: pass
            # base-fallback
            base=self.deref(label); seq2=self.TDF_LabelSequence()
            try:
                self.shape_tool.GetComponents_s(base, seq2)
                for i in range(1,seq2.Length()+1): yield seq2.Value(i)
            except: return
        def get_components_list(self,label:ProjectDialog.TDF_Label)->List[ProjectDialog.TDF_Label]:
            """
            Return all component labels for a given label as a list.

            Steps:
                1. Try instance-first components via GetComponents_s(label, seq).
                2. If none are found, dereference the label and try GetComponents_s(base, seq2).
                3. Collect all found component labels into a Python list.
                4. Return the resulting list, which may be empty if no components exist.

            Args:
                label (TDF_Label): The label for which to list components.

            Returns:
                list[TDF_Label]: A list of component labels; empty if none.

            Notes:
                Uses self.deref() and XCAFDoc_ShapeTool.GetComponents_s.
            """
            out=[]
            seq=self.TDF_LabelSequence()
            try:
                self.shape_tool.GetComponents_s(label, seq)
                for i in range(1,seq.Length()+1): out.append(seq.Value(i))
            except: pass
            base=self.deref(label)
            if not out and base != label:
                seq2=self.TDF_LabelSequence()
                try:
                    self.shape_tool.GetComponents_s(base, seq2)
                    for i in range(1,seq2.Length()+1): out.append(seq2.Value(i))
                except: pass
            return out
        def get_all_shapes(self)->Iterable[ProjectDialog.TDF_Label]:
            """
            Yield all shape labels known to the XCAF shape tool.

            Steps:
                1. Create a TDF_LabelSequence for shape labels.
                2. Call ShapeTool.GetShapes(seq); if that fails, recreate a shape
                tool from the document main label and call GetShapes again.
                3. Yield each label in the resulting sequence.

            Args:
                None

            Returns:
                Iterable[TDF_Label]: An iterator of all shape labels in the document.

            Notes:
                Uses XCAFDoc_ShapeTool.GetShapes and ShapeTool_s(self.doc.Main()).
            """
            seq=self.TDF_LabelSequence()
            try:
                self.shape_tool.GetShapes(seq)
            except Exception:
                try:
                    self.XCAFDoc_DocumentTool.ShapeTool_s(self.doc.Main()).GetShapes(seq)
                except Exception:
                    seq=self.TDF_LabelSequence()
            for i in range(1,seq.Length()+1): yield seq.Value(i)
        def is_assembly(self,label:ProjectDialog.TDF_Label)->bool:
            """
            Determine whether a label represents an assembly.

            Steps:
                1. Dereference the label to its base definition.
                2. Call ShapeTool.IsAssembly_s(base).
                3. Convert the result to bool; return False on failure.

            Args:
                label (TDF_Label): The label to test.

            Returns:
                bool: True if the dereferenced label is an assembly; False otherwise.

            Notes:
                Uses self.deref() and XCAFDoc_ShapeTool.IsAssembly_s.
            """
            try: return bool(self.shape_tool.IsAssembly_s(self.deref(label)))
            except: return False
        def is_leaf(self,label:ProjectDialog.TDF_Label)->bool:
            """
             Check if a label is a leaf part with at least one solid body.

            Steps:
                1. Dereference the label to its base definition.
                2. If the base is an assembly, return False.
                3. If the label has components (children), return False.
                4. Get the base shape via ShapeTool.GetShape_s.
                5. Use TopExp_Explorer to test if there is at least one TopAbs_SOLID.
                6. Return True only if at least one solid is found and there are no children.

            Args:
                label (TDF_Label): The label to test.

            Returns:
                bool: True if the label is a leaf with solids; False otherwise.

            Notes:
                Uses self.deref(), self.get_components_list(), ShapeTool.GetShape_s,
                and TopExp_Explorer over TopAbs_SOLID.
            """
            base=self.deref(label)
            try:
                if bool(self.shape_tool.IsAssembly_s(base)): return False
            except: pass
            if self.get_components_list(label):
                return False
            try: shp=self.shape_tool.GetShape_s(base)
            except: return False
            try:
                exp=self.TopExp_Explorer(shp,self.TopAbs_SOLID); 
                if exp.More(): return True
            except: pass
            return False
        def get_shape(self,label:ProjectDialog.TDF_Label):
            """
            Get the TopoDS_Shape associated with a label, after dereferencing.

            Steps:
                1. Dereference the label to its base/definition label.
                2. Call ShapeTool.GetShape_s(base) to obtain the shape.
                3. Return the resulting TopoDS_Shape.

            Args:
                label (TDF_Label): The label whose shape is requested.

            Returns:
                TopoDS_Shape: The shape associated with the label, or a null shape
                if none is assigned.

            Notes:
                Uses self.deref() and XCAFDoc_ShapeTool.GetShape_s.
            """

            return self.shape_tool.GetShape_s(self.deref(label))
        def iter_tree(self,label:ProjectDialog.TDF_Label,path:str="")->Iterable[Tuple[ProjectDialog.TDF_Label,str]]:
            """
            Iterate recursively over the assembly tree, yielding labels and their paths.

            Steps:
                1. Build the current path by appending this label's name to the
                incoming path.
                2. Yield a tuple of (label, path_string) for the current node.
                3. Retrieve all components (children) of the label.
                4. Recursively call iter_tree on each child, passing the updated path.

            Args:
                label (TDF_Label): The starting label (assembly or part).
                path (str): Current path prefix, built as the traversal descends.

            Returns:
                Iterable[Tuple[TDF_Label, str]]: Iterator over (label, path) pairs
                representing the hierarchical structure.

            Notes:
                Uses self.get_name() and self.get_components_list().
            """
            here=(path+"/"+self.get_name(label)) if path else self.get_name(label)
            yield (label,here)
            for child in self.get_components_list(label):
                yield from self.iter_tree(child,here)
        def iter_part_solids(self, label):
            """Iterate solids under a part label and attempt to map each solid back to a sublabel.

            Steps:
                1. Dereference the incoming label to its base/definition label.
                2. Retrieve the overall part shape via get_shape(label).
                3. Look up a FindSubShape function on the shape tool, supporting
                both _s and non-_s signatures.
                4. Use TopExp_Explorer to iterate all TopAbs_SOLID sub-shapes.
                5. For each solid:
                    a. Try FindSubShape(base, solid) to find a corresponding sublabel.
                    b. If that fails, try FindSubShape(solid, out_label) variant.
                    c. Yield (solid_shape, solid_label_or_None, running_index).

            Args:
                label (TDF_Label): The part label whose solids are to be enumerated.

            Returns:
                Iterable[Tuple[TopoDS_Shape, Optional[TDF_Label], int]]:
                    An iterator of (solid shape, associated label or None, 1-based index).

            Notes:
                Uses self.deref(), self.get_shape(), XCAFDoc_ShapeTool.FindSubShape,
                and TopExp_Explorer over TopAbs_SOLID.
            """
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_SOLID
            from OCP.TDF import TDF_Label

            base = self.deref(label)
            shp  = self.get_shape(label)

            # FindSubShape can be exposed with or without _s and with two signatures
            find = getattr(self.shape_tool, "FindSubShape_s", None) or getattr(self.shape_tool, "FindSubShape", None)

            idx = 0
            xp = TopExp_Explorer(shp, TopAbs_SOLID)
            while xp.More():
                bod = xp.Current()
                xp.Next()
                idx += 1

                slab = None
                if callable(find):
                    # Try (parent_label, shape) -> label
                    try:
                        slab = find(base, bod)
                        if hasattr(slab, "IsNull") and slab.IsNull():
                            slab = None
                    except TypeError:
                        # Try (shape, out_label) -> bool
                        tmp = TDF_Label()
                        try:
                            ok = bool(find(bod, tmp))
                            if ok and not (hasattr(tmp, "IsNull") and tmp.IsNull()):
                                slab = tmp
                        except Exception:
                            pass
                yield (bod, slab, idx)

    def _empty_node(self, name: str) -> dict:
        """
        Create a new, initialized node dictionary for the JSON structure.

        Steps:
            1. Generate a new UUID string for the node's id.
            2. Fill in the common fields (name, volume, density, material,
               mass, total mass, components) with default values.
            3. Return the resulting dictionary.

        Args:
            name (str): Display name for the node (assembly, part, or solid).

        Returns:
            dict: A node dictionary with default numeric fields set to None
            and an empty components list.

        Notes:
            Uses uuid.uuid4() to generate a unique id.
        """
        return {
            "id": str(uuid.uuid4()),
            "name": name,
            "volume [mm^3]": None,
            "density [kg/mm^3]": None,
            "material": None,
            "mass [kg]": None,
            "total mass [kg]": None,
            "components": []
        }

    def build_tree_node(
        self,
        xde,
        label,
        base_unit: str,
        collapse_single_solid: bool = True,
        density_kg_per_mm3: float | None = DEFAULT_DENSITY_KG_PER_MM3,
    ):
        """
        Build a single node (assembly or part) of the hierarchical JSON tree.

        Steps:
            1. Resolve the label's name and create a base node dictionary via _empty_node().
            2. Query the label's children via xde.get_components_list(label).
            3. If children exist (assembly):
                a. Recursively call build_tree_node() for each child.
                b. Attach non-None child nodes to the components list.
                c. Return the node if it has any components; otherwise return None.
            4. If no children (leaf part):
                a. Use xde.iter_part_solids(label) to find all solids and their volumes.
                b. Convert each solid volume from base_unit³ to mm³ using vol_convert().
                c. If no solids are found, return None (e.g. COMPOUND placeholders).
                d. If there is exactly one solid and collapse_single_solid is True:
                    - Store its volume and mass directly on the part node,
                      clear components, and return the node.
                e. If there are multiple solids:
                    - Create a child node per solid with its own volume and mass,
                      and append these to the components list.
                f. Return the node.

        Args:
            xde (XDE): The XDE helper instance used for shape and name queries.
            label (TDF_Label): The label representing the current assembly or part.
            base_unit (str): The unit in which the raw volume was computed (e.g. "mm").
            collapse_single_solid (bool): If True, collapse a single-solid part so
                that volume and mass live on the part node instead of a child solid node.
            density_kg_per_mm3 (float | None): Uniform density used to compute mass;
                if None, no mass is computed.

        Returns:
            dict | None: The constructed node dictionary, or None if the label
            should be skipped (e.g. no solids and no children).

        Notes:
            Uses self._empty_node(), xde.get_name(), xde.get_components_list(),
            xde.iter_part_solids(), and self.vol_convert().
        """

        name = xde.get_name(label).strip()
        node = self._empty_node(name)

        # Assembly (has components) -> recurse
        children = xde.get_components_list(label)
        if children:
            for ch in children:
                child_node = self.build_tree_node(
                    xde,
                    ch,
                    base_unit,
                    collapse_single_solid=collapse_single_solid,
                    density_kg_per_mm3=density_kg_per_mm3,
                )
                if child_node is not None:
                    node["components"].append(child_node)
            return node if node["components"] else None

        # Leaf: gather solids
        solids: list[tuple[str, float]] = []  # (solid_name, volume_mm3)
        for solid_shape, solid_label, sidx in xde.iter_part_solids(label):
            sname = xde.get_name(solid_label) if solid_label else f"{name} [solid {sidx}]"

            # volume in base units -> mm^3
            try:
                v_b = self.cq.Shape(solid_shape).Volume()
            except Exception:
                v_b = 0.0
            v_mm3 = self.vol_convert(v_b, base_unit, "mm")
            solids.append((sname, v_mm3))

        if not solids:
            # e.g. COMPOUND placeholder with no solids
            return None

        # ---- collapse case: one solid -> put volume & mass on the part node ----
        if collapse_single_solid and len(solids) == 1:
            sname, v_mm3 = solids[0]
            node["volume [mm^3]"] = v_mm3
            if density_kg_per_mm3 is not None:
                node["density [kg/mm^3]"] = density_kg_per_mm3
                m = v_mm3 * density_kg_per_mm3
                node["mass [kg]"] = m
                node["total mass [kg]"] = m  # leaf: self == total
            node["components"] = []
            return node

        # ---- multiple solids -> emit children, each with its own volume & mass ----
        for sname, v_mm3 in solids:
            snode = self._empty_node(sname)
            snode["volume [mm^3]"] = v_mm3
            if density_kg_per_mm3 is not None:
                snode["density [kg/mm^3]"] = density_kg_per_mm3
                m = v_mm3 * density_kg_per_mm3
                snode["mass [kg]"] = m
                snode["total mass [kg]"] = m  # leaf: self == total
            node["components"].append(snode)

        return node

    def build_tree_json(self, xde, base_unit: str) -> list:
        """
        Build the full hierarchical JSON structure for all free shapes in the STEP model.

        Steps:
            1. Initialize an empty list for the top-level nodes.
            2. Iterate over all free shape labels from the XDE helper.
            3. For each root label, call build_tree_node() to construct its
               subtree (assemblies, parts, and solids).
            4. Append any non-None root nodes to the list.
            5. Return the final list of root node dictionaries.

        Args:
            xde (XDE): The XDE helper instance for querying labels and shapes.
            base_unit (str): The base unit used for volume computation (e.g. "mm").

        Returns:
            list[dict]: A list of root node dictionaries representing the full
            assembly tree.

        Notes:
            Uses xde.get_free_shapes() and self.build_tree_node().
        """
        out = []
        for root in xde.get_free_shapes():
            n = self.build_tree_node(
                xde,
                root,
                base_unit,
                collapse_single_solid=True,
                density_kg_per_mm3=self.DEFAULT_DENSITY_KG_PER_MM3,
            )
            if n is not None:
                out.append(n)
        return out

    def accumulate_volume_and_mass(self, node: dict) -> tuple[float, float]:
        """
        Aggregate volume and mass up the tree, filling part/assembly totals.

        Steps:
            1. Read the node's own volume and mass (or treat None as 0.0).
            2. If the node has no components (leaf):
                a. Set total mass to its own mass, if non-zero.
                b. Return (volume, mass).
            3. If the node has children:
                a. Recursively call accumulate_volume_and_mass() on each child.
                b. Sum child volumes and child masses.
                c. If the node has no own volume, set it to the sum of child volumes.
                d. If the node has no own mass, set it to the sum of child masses.
                e. Set total mass to the sum of child masses (or node mass if children have none).
            4. Return the node's effective (volume, mass).

        Args:
            node (dict): A node dictionary representing an assembly, part, or solid.

        Returns:
            tuple[float, float]: A tuple (volume_mm3, mass_kg) for this node,
            reflecting its own values or aggregated child values.

        Notes:
            Calls itself recursively to walk the entire tree in post-order.
        """
        vol_self = node["volume [mm^3]"] or 0.0
        mass_self = node["mass [kg]"] or 0.0

        comps = node.get("components") or []
        if not comps:
            # leaf (solid or collapsed single-solid part)
            node["total mass [kg]"] = mass_self if mass_self != 0.0 else node.get("total mass [kg]")
            return vol_self, mass_self

        total_child_vol = 0.0
        total_child_mass = 0.0
        for ch in comps:
            cv, cm = self.accumulate_volume_and_mass(ch)
            total_child_vol += cv
            total_child_mass += cm

        # if this node has no own volume, adopt sum of children
        if (node["volume [mm^3]"] is None or node["volume [mm^3]"] == 0.0) and total_child_vol != 0.0:
            node["volume [mm^3]"] = total_child_vol
            vol_self = total_child_vol

        # if this node has no own mass, adopt sum of children
        if (node["mass [kg]"] is None or node["mass [kg]"] == 0.0) and total_child_mass != 0.0:
            node["mass [kg]"] = total_child_mass
            mass_self = total_child_mass

        # total mass = whole subtree
        if total_child_mass != 0.0:
            node["total mass [kg]"] = total_child_mass
        elif mass_self != 0.0:
            node["total mass [kg]"] = mass_self

        return vol_self, mass_self

    def process_step_file(self):
        """
        Load the STEP file, build the assembly/part/solid tree, and aggregate volumes and masses.

        Steps:
        1.  Try to create an XDE model from `self.step_file_path` with
            `prefer_definition_name=True`, so names are taken from definition/base
            labels rather than instance labels.
        2.  Fix the geometric base unit to millimetres (`base_unit = "mm"`).
        3.  Build the hierarchical structure of assemblies, parts, and solids by
            calling `self.build_tree_json(xde, base_unit)` and store it in
            `self.complete_structure`.
        4.  For each top-level node in `self.complete_structure`, call
            `self.accumulate_volume_and_mass(node)` to propagate volumes and masses
            from solids up to their parent parts and assemblies.

        Args:
            None

        Returns:
            None

        Error Handling:
            - If loading fails, log the exception and return error code 3.

        Notes:
            - The method relies on several helper methods, including `build_tree_json` and
            `accumulate_volume_and_mass`.
        """
        
        try:
            xde=self.XDE(self.step_file_path, prefer_definition_name=True)
        except Exception as e:
            self.LOG.exception("Failed to read STEP: %s", e); return 3
        base_unit = "mm"
        self.complete_structure = self.build_tree_json(xde, base_unit)
        for node in self.complete_structure:
            self.accumulate_volume_and_mass(node)
        
        
        # Show the structure in the tree widget
        self.populate()
        self.tree.expandAll()

    def find_matches(self, text: str, pattern: str):
        """
        Find matches in a given text based on the provided regex pattern.

        This method uses the `re.findall` function to search for all occurrences
        of the specified regex pattern in the provided text.

        Args:
            text (str): The input text to search within.
            pattern (str): The regex pattern to use for matching.

        Returns:
            list: A list of all matches found in the text. Each match is represented
                as a string or tuple, depending on the regex pattern.

        Notes:
            - The search is performed in DOTALL mode, meaning the `.` in the regex
            pattern matches newline characters as well.
        """
        return re.findall(pattern, text, re.DOTALL)
    
    def prep_step_file(self, content: str, ):
        """
        Prepare the STEP file content for processing.

        This method preprocesses the content of a STEP file by removing unnecessary
        line breaks and whitespace, and by formatting the content to make it easier
        to parse. It splits the processed content into individual lines for further
        analysis.

        Args:
            content (str): The raw content of the STEP file as a string.

        Returns:
            list: A list of strings, where each string represents a line in the
                preprocessed STEP file.

        Notes:
            - Unnecessary line breaks and excess whitespace are removed.
            - A line break is inserted after every occurrence of `");` to improve readability.
        """
        # Remove unnecessary line breaks within specific patterns
        content = re.sub(r"\s+", "", content)

        # Insert a line break after every occurrence of '");'
        content = re.sub(r'\);', ');\n', content)

        lines = content.splitlines()

        return lines
    
    def get_info_from_step(self, lines, DENSITY_PATTERN, NAME_MATERIAL_PATTERN):

        """
        Extract volumes, densities, names, and materials from the prepared STEP file.

        This method processes the lines of a STEP file and uses regex patterns to extract
        information about volumes, densities, names, and materials. The extracted data is
        returned as separate lists, each containing tuples of the extracted value and the
        corresponding line number.

        Args:
            lines (list): A list of strings, where each string represents a line in the
                        preprocessed STEP file.
            VOLUME_PATTERN (str): The regex pattern to extract volume information.
            DENSITY_PATTERN (str): The regex pattern to extract density information.
            NAME_MATERIAL_PATTERN (str): The regex pattern to extract names and materials.

        Returns:
            tuple: A tuple containing four lists:
                - volumes (list): A list of tuples (volume, line_number).
                - densities (list): A list of tuples (density, line_number).
                - names (list): A list of tuples (name, line_number).
                - materials (list): A list of tuples (material, line_number).

        Notes:
            - The method assumes that the `NAME_MATERIAL_PATTERN` can distinguish between
            names and materials based on whether the two captured groups are equal.
            - If the groups are equal, the value is treated as a name; otherwise, it is
            treated as a material.
        """

        #volumes = []
        densities = []
        names = []
        materials = []

        # Extract volumes, densities, and names with line numbers
        for line_number, line in enumerate(lines):
        #    volume_match = re.search(VOLUME_PATTERN, line)
            density_match = re.search(DENSITY_PATTERN, line)
            name_material_match = re.search(NAME_MATERIAL_PATTERN, line)

        #    if volume_match:
        #        volumes.append((float(volume_match.group(1)), line_number))
            if density_match:
                densities.append((float(density_match.group(1)), line_number))
            if name_material_match and name_material_match.group(1) == name_material_match.group(2):
                names.append((name_material_match.group(1), line_number))
            elif name_material_match and name_material_match.group(1) != name_material_match.group(2):
                materials.append((name_material_match.group(1), line_number))

        return densities, names, materials

    def build_structure(self, hierarchy, updated_node_map): 
        """
        Build a complete hierarchical structure with enriched data.

        This method processes the given hierarchy and enriches each node with additional
        data such as volume, density, material, mass, and total mass. It recursively
        calculates the total mass and volume for each node based on its child components.

        Args:
            hierarchy (list): A list of dictionaries representing the hierarchical structure
                            of components.
            updated_node_map (dict): A dictionary containing updated data for nodes, where
                                    the keys are node IDs and the values are dictionaries
                                    with additional data (e.g., volume, density, material).

        Returns:
            list: A list of enriched nodes representing the complete hierarchical structure.

        Notes:
            - The `enrich_node` function is used recursively to process each node and its
            child components.
            - Mass is calculated as `mass = volume * density` if both volume and density
            are available.
            - Total mass is calculated as the sum of the masses of all child components.
            - Total volume is calculated as the sum of the volumes of all child components.
            - If a node has no child components, its total mass and total volume are set
            to its own mass and volume, respectively.
        """
        def enrich_node(tree):
            node_id = tree['id']
            data = updated_node_map.get(node_id, {})
            volume = data.get('volume [mm^3]', tree.get('volume [mm^3]'))
            density = data.get('density [kg/mm^3]', tree.get('density [kg/mm^3]'))
            material = data.get('material', tree.get('material'))

            # Calculate mass: mass = volume * density
            if volume is not None and density is not None:
                mass = volume * density
            else:
                mass = None

            # Recursively enrich components and calculate their mass
            components = [enrich_node(child) for child in tree.get('components', [])]

            if components:
                # Calculate total mass: sum of 'total mass [kg]' or 'mass [kg]' of components
                total_mass = sum(
                    child.get('total mass [kg]', child.get('mass [kg]', 0)) 
                    for child in components 
                    if child.get('total mass [kg]') is not None or child.get('mass [kg]') is not None
                )
                # Calculate total volume: sum of volumes of components
                total_volume = sum(child['volume [mm^3]'] for child in components if child['volume [mm^3]'] is not None)
            else:
                total_mass = mass
                total_volume = volume

            enriched = {
                'id': node_id,
                'name': tree['name'],
                'volume [mm^3]': total_volume,
                'density [kg/mm^3]': density,
                'material': material,
                'mass [kg]': mass,
                'total mass [kg]': total_mass,
                'components': components
            }
            return enriched

        self.complete_structure = [enrich_node(tree) for tree in hierarchy]
        return self.complete_structure

    def build_hierarchy(self, usage_matches):
        """
        Build a hierarchical structure from parent-child relationships.

        This method processes a list of parent-child relationships and constructs a
        hierarchical tree structure. Each node in the hierarchy is assigned a unique
        identifier and can have child components. Additionally, a mapping of node IDs
        to their corresponding data is created.

        Args:
            structure_matches (list): A list of tuples representing parent-child relationships,
                                    where each tuple is of the form (child, parent).

        Returns:
            tuple: A tuple containing:
                - hierarchy (list): A list of dictionaries representing the hierarchical structure
                                    of components, starting from the root nodes.
                - node_map (dict): A dictionary mapping node IDs to their corresponding data,
                                including name, volume, density, material, and child components.

        Notes:
            - Root nodes are identified as nodes that do not appear as children in the
            `structure_matches`.
            - Each node is assigned a unique ID using the `uuid` module.
            - The `map_nodes` function is used recursively to populate the `node_map` with
            node data.
        """
        step_text = read_file(self.step_file_path)
        if step_text is None:
            return

        # STEP ENTITY RESOLUTION: DEF -> FORMATION -> PRODUCT -> NAME
        #def resolve_def_names(text):
        product_names = dict(self.find_matches(step_text, self.NAMES_PATTERN))
        formation_to_product = dict(self.find_matches(step_text, self.FORMATION_TO_PRODUCT_PATTERN))
        definition_to_formation = dict(self.find_matches(step_text, self.DEFINITION_TO_FORMATION_PATTERN))

        def_to_name = {}
        for def_id, formation_id in definition_to_formation.items():
            product_id = formation_to_product.get(formation_id)
            name = product_names.get(product_id, "Unnamed")
            def_to_name[def_id] = name
            
        # Build parent-child mapping
        children_map = defaultdict(list)
        parents = set()
        children = set()
    
        for parent_def, child_def in usage_matches:
            children_map[parent_def].append(child_def)
            parents.add(parent_def)
            children.add(child_def)
    
        # Find root nodes (those without parents)
        root_nodes = parents - children

        # Function to recursively build the tree
        def build_tree(node):
            unique_id = str(uuid.uuid4())
            part_name = def_to_name.get(node, "Unnamed")
            tree = {
                'id': unique_id,
                'name': part_name,
                'components': []
            }
            for child in children_map.get(node, []):
                tree['components'].append(build_tree(child))
            return tree
    
        hierarchy = []
        node_map = {}
        for root in root_nodes:
            hierarchy.append(build_tree(root))
    
        # Map from ID to node
        def map_nodes(tree, node_map):
            node_map[tree['id']] = {
                'name': tree['name'],
                'volume [mm^3]': None,
                'density [kg/mm^3]': None,
                'material': None,
                'components': [child['id'] for child in tree['components']]
            }
            for child in tree['components']:
                map_nodes(child, node_map)
    
        for root in hierarchy:
            map_nodes(root, node_map)
    
        return hierarchy, node_map

    def assign_info(self, volumes, densities, names, materials, node_map): 
        """
        Assign extracted information (volumes, densities, materials) to nodes in the node map.

        This method processes the extracted data from the STEP file and assigns the closest
        volume, density, and material information to each node in the `node_map` based on
        their corresponding names and line numbers.

        Args:
            volumes (list): A list of tuples (volume, line_number) extracted from the STEP file.
            densities (list): A list of tuples (density, line_number) extracted from the STEP file.
            names (list): A list of tuples (name, line_number) extracted from the STEP file.
            materials (list): A list of tuples (material, line_number) extracted from the STEP file.
            node_map (dict): A dictionary mapping node IDs to their corresponding data, including
                            name, volume, density, and material.

        Returns:
            dict: The updated `node_map` with assigned volume, density, and material information.

        Notes:
            - For each name in the `names` list, the method identifies the corresponding nodes
            in the `node_map` and assigns the closest volume, density, and material based on
            their line numbers.
            - If multiple volumes are found after a name, the closest density and material are
            determined based on their line number proximity to the name.
            - If no matching data is found, the fields are set to `None`.
        """

        for i, (name, line_num) in enumerate(names):
            volumes_after_name = []
            # Find the component IDs with this name
            component_ids = [cid for cid, info in node_map.items() if info["name"] == name]
        
            for component_id in component_ids:
                # Reset fields to None
                node_map[component_id].update({
                    "volume [mm^3]": None,
                    "density [kg/mm^3]": None,
                    "material": None
                })

                # Determine the range for the next data
                if i < len(names) - 1:
                    next_name_line_num = names[i + 1][1]
                else:
                    next_name_line_num = float('inf')
  
                # Collect volumes after the current name
                for volume, v_line_num in volumes:
                    if line_num < v_line_num < next_name_line_num:
                        volumes_after_name.append(volume)

                # Assign the closest volume
                closest_volume = volumes_after_name[0] if volumes_after_name else None

                if len(volumes_after_name) >= 2:
                    # Find the closest density
                    min_density_delta = float('inf')
                    closest_density = None
                    for density, d_line_num in densities:
                        delta = abs(line_num - d_line_num)
                        if delta < min_density_delta:
                            min_density_delta = delta
                            closest_density = density

                    # Find the closest material
                    min_material_delta = float('inf')
                    closest_material = None
                    for material, m_line_num in materials:
                        delta = abs(line_num - m_line_num)
                        if delta < min_material_delta:
                            min_material_delta = delta
                            closest_material = material
                else:
                    # If not enough data, set to None
                    closest_density = None
                    closest_material = None

                # Assign the information to the component
                node_map[component_id]["volume [mm^3]"] = closest_volume
                node_map[component_id]["density [kg/mm^3]"] = closest_density
                node_map[component_id]["material"] = closest_material        

        return node_map
    
    
    def save_hierarchy(self):
        """
        Save the current assembly and component data back to a JSON file.

        This method creates a structured JSON object containing the project name, version,
        and the complete hierarchical structure of components. It then saves this data to
        a JSON file with a filename based on the project name and version.

        Args:
            None

        Returns:
            None

        Notes:
            - The JSON file is saved in the current working directory.
            - The filename is generated using the project name and version in the format:
            `<project_name>_<version>_structure.json`.
            - The `dump_file` function is used to write the JSON data to the file.
        """
        
        # Create a new structure to include project name and version
        self.output_structure = {
            "project_name": self.project_name_input.text(),
            "version": self.version_input.text(),
            "data": self.complete_structure
        }
        
        self.output_path = f"{self.project_name_input.text()}_" \
                           f"{self.version_input.text()}_structure.json"
        dump_file(self.output_structure, self.output_path)
        self.project_data = self.output_structure

    def save_all(self): 
        """
        Save all data and close the dialog.

        This method saves the current hierarchical structure and project data to a JSON file
        by calling the `save_hierarchy` method. After saving, it closes the dialog by calling
        the `accept` method.

        Args:
            None

        Returns:
            None

        Notes:
            - This method ensures that all changes made to the project are persisted before
            the dialog is closed.
        """
        self.save_hierarchy()
        self.accept()
    
    def on_item_changed(self, item: QTreeWidgetItem, column: int):
        """
        React to inline edits in Volume (1), Density (2), or Material (3).
        When Volume or Density changes, Mass is recalculated immediately and
        the result is shown in column 4. For leaf nodes, Total Mass is also
        updated. The underlying node in complete_structure is kept in sync.

        Args:
            item (QTreeWidgetItem): The item that was edited.
            column (int): The column index that changed.
        """
        if self._updating_tree or column not in (1, 2, 3):
            return

        node_id = item.data(0, Qt.UserRole)
        if not node_id:
            return

        try:
            volume   = float(item.text(1)) if item.text(1).strip() else None
            density  = float(item.text(2)) if item.text(2).strip() else None
            material = item.text(3).strip() or None

                        # Recalculate mass
            if volume is not None and density is not None:
                mass     = volume * density
                mass_str = f"{round(mass, 3)}"
            else:
                mass     = None
                mass_str = ""

            # Write back to the tree without triggering another signal
            self._updating_tree = True
            item.setText(4, mass_str)
            if not item.childCount():          # leaf: own mass == total mass
                item.setText(5, mass_str)
            self._updating_tree = False

            # Persist to the data structure
            self._update_node_in_structure(node_id, {
                "volume [mm^3]":    volume,
                "density [kg/mm^3]": density,
                "material":          material,
                "mass [kg]":         mass,
                "total mass [kg]":   mass if not item.childCount() else None,
            })

        except ValueError:
            pass   # user is still typing — ignore transient bad input

    def _find_node_by_id(self, structure, node_id: str) -> dict | None:
        """
        Recursively search complete_structure for the node with the given id.

        Args:
            structure (list | dict): The subtree to search.
            node_id (str): The UUID to look for.

        Returns:
            dict | None: The matching node dict, or None if not found.
        """
        if isinstance(structure, list):
            for item in structure:
                result = self._find_node_by_id(item, node_id)
                if result is not None:
                    return result
        elif isinstance(structure, dict):
            if structure.get("id") == node_id:
                return structure
            for child in structure.get("components", []):
                result = self._find_node_by_id(child, node_id)
                if result is not None:
                    return result
        return None

    def _update_node_in_structure(self, node_id: str, data: dict):
        """
        Update specific fields of the node identified by node_id.

        Only non-None values overwrite existing data, except for 'material'
        which may legitimately be set to None to clear it.

        Args:
            node_id (str): UUID of the node to update.
            data (dict): Key-value pairs to apply.
        """
        node = self._find_node_by_id(self.complete_structure, node_id)
        if node is None:
            return
        for key, value in data.items():
            if value is not None or key == "material":
                node[key] = value

    def edit_component(self):
        """
        Open EditComponentDialog for the selected tree item.

        Lets the user change Density and Material; Mass is recalculated live
        in the dialog. On acceptance the node and all parent totals are updated.

        Error Handling:
            Warns if nothing is selected or the node cannot be located.
        
        """
        selected = self.tree.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a component to edit.")
            return

        item    = selected[0]
        node_id = item.data(0, Qt.UserRole)
        if not node_id:
            QMessageBox.warning(self, "Warning", "Invalid selection.")
            return

        node = self._find_node_by_id(self.complete_structure, node_id)
        if node is None:
            QMessageBox.warning(self, "Warning", "Component not found in structure.")
            return

        dialog = EditComponentDialog(self, node_data=node)
        if dialog.exec() == QDialog.Accepted:
            node.update(dialog.get_data())
            # Re-propagate totals from every root downward
            for root_node in self.complete_structure:
                self.accumulate_volume_and_mass(root_node)
            self.populate()
            self.tree.expandAll()
            QMessageBox.information(self, "Success", "Component updated successfully.")

class AddComponentDialog(QDialog):

    def __init__(self, parent=None, assemblies=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Component")
        self.setMinimumWidth(420)
        self.setModal(True)
        self.materials = {}  # Will be populated in _build_ui()
        self._build_ui(assemblies or [])

    def _load_materials(self):
        """
        Load materials from the JSON file and populate the material dropdown.
        Uses config.materials_json if available; otherwise falls back to 'materials_validation.json'.
        """
        parent = self.parent()
        if hasattr(parent, 'config') and parent.config:
            materials_path = parent.config.materials_json
        else:
            materials_path = "app_data/materials_validation.json"

        try:
            with open(materials_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.materials = data.get("materials", {})
        except Exception as e:
            print(f"Materials could not be loaded: {e}")
            self.materials = {}

        # Populate dropdown: sort alphabetically, add placeholder item
        self.material_dropdown.clear()
        self.material_dropdown.addItem("Select material…", None)
        for mat_name in sorted(self.materials.keys()):
            self.material_dropdown.addItem(mat_name, mat_name)

    def _build_ui(self, assemblies):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        form.setLabelAlignment(Qt.AlignRight)

        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("e.g. Bracket_01")
        form.addRow("Name:", self.name_input)

        self.volume_input = QLineEdit(self)
        self.volume_input.setPlaceholderText("e.g. 12500.0")
        self.volume_input.textChanged.connect(self._refresh_mass)
        form.addRow("Volume [mm³]:", self.volume_input)

        # populated from JSON
        self.material_dropdown = QComboBox(self)
        self._load_materials()
        self.material_dropdown.currentIndexChanged.connect(self._on_material_changed)
        form.addRow("Material:", self.material_dropdown)
        
        # read-only, auto-updated
        self.density_input = QLineEdit(self)
        self.density_input.setReadOnly(True)
        self.density_input.setText("—")
        form.addRow("Density [kg/mm³]:", self.density_input)

        self.mass_label = QLabel("—")
        form.addRow("Mass [kg]:", self.mass_label)

        self.assembly_dropdown = QComboBox(self)
        self.assembly_dropdown.addItems(assemblies)
        form.addRow("Assign to Assembly:", self.assembly_dropdown)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_material_changed(self, index):
        """
        Called when the user selects a material from the dropdown.
        Populates the density field and recalculates mass.
        """
        if index <= 0:
            # Placeholder selected
            self.density_input.setText("—")
            self._refresh_mass()
            return

        mat_key = self.material_dropdown.currentData()  # e.g. "STEEL_1.4526"
        mat_data = self.materials.get(mat_key, {})
        density = mat_data.get("density [kg/mm^3]")

        if density is not None:
            self.density_input.setText(str(density))
        else:
            self.density_input.setText("N/A")

        self._refresh_mass()

    def _refresh_mass(self):
        """Recalculate and display mass whenever volume or density changes."""
        try:
            vol_text = self.volume_input.text().strip()
            den_text = self.density_input.text().strip()

            vol = float(vol_text) if vol_text else None
            den = float(den_text) if (den_text and den_text not in ("—", "N/A")) else None

            if vol is not None and den is not None:
                self.mass_label.setText(f"{round(vol * den, 6)} kg")
                self.mass_label.setStyleSheet("color: black;")
            else:
                self.mass_label.setText("—")
                self.mass_label.setStyleSheet("")
        except ValueError:
            self.mass_label.setText("Invalid input")
            self.mass_label.setStyleSheet("color: red;")

    def get_data(self) -> dict:
        volume  = float(self.volume_input.text()) if self.volume_input.text().strip() else 0.0
        den_text = self.density_input.text().strip()
        density = float(den_text) if (den_text and den_text not in ("—", "N/A")) else 0.0
        mass    = volume * density
        mat_key = self.material_dropdown.currentData()  # Store material key, not display name

        return {
            'id':               str(uuid.uuid4()),
            'name':             self.name_input.text().strip(),
            'volume [mm^3]':    volume,
            'density [kg/mm^3]': density,
            'material':         mat_key,  # key from JSON, not free text
            'mass [kg]':        mass,
            'total mass [kg]':  mass,
            'components':       [],
            'assigned_assembly': self.assembly_dropdown.currentText(),
        }
    
class EditComponentDialog(QDialog):
    
    def __init__(self, parent=None, node_data: dict | None = None):
        super().__init__(parent)
        self.node_data = node_data or {}
        self.setWindowTitle("Edit Component")
        self.setMinimumWidth(420)
        self.materials = {}  # Will be populated in _build_ui()
        self._build_ui()

    def _load_materials(self):
        """
        Load materials from JSON and populate the dropdown.
        Pre-selects the current material from node_data if available.
        """
        parent = self.parent()
        if hasattr(parent, 'config') and parent.config:
            materials_path = parent.config.materials_json
        else:
            materials_path = "app_data/materials_validation.json"

        try:
            with open(materials_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.materials = data.get("materials", {})
        except Exception as e:
            print(f"Materials could not be loaded: {e}")
            self.materials = {}

        # Populate dropdown
        self.material_dropdown.clear()
        self.material_dropdown.addItem("Select material…", None)
        for mat_name in sorted(self.materials.keys()):
            self.material_dropdown.addItem(mat_name, mat_name)

        # Pre-select current material
        current_mat = self.node_data.get("material")
        if current_mat:
            idx = self.material_dropdown.findData(current_mat)
            if idx >= 0:
                self.material_dropdown.setCurrentIndex(idx)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        form.setLabelAlignment(Qt.AlignRight)

        # Editable name
        self.name_input = QLineEdit(self)
        self.name_input.setText(str(self.node_data.get("name", "")))
        form.addRow("Name:", self.name_input)

        # Read-only volume (context only)
        vol     = self.node_data.get("volume [mm^3]")
        vol_str = f"{round(vol, 6)}" if vol is not None else "N/A"
        form.addRow("Volume [mm³]:", QLabel(vol_str))

        # read-only, auto-updated
        self.density_input = QLineEdit(self)
        self.density_input.setReadOnly(True)
        self.density_input.setText("—")
        form.addRow("Density [kg/mm³]:", self.density_input)

        # Material dropdown
        self.material_dropdown = QComboBox(self)
        self._load_materials()
        self.material_dropdown.currentIndexChanged.connect(self._on_material_changed)
        form.addRow("Material:", self.material_dropdown)

        # Live mass (read-only)
        self.mass_label = QLabel("—")
        form.addRow("Mass [kg]:", self.mass_label)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Initialize density and mass
        self._on_material_changed(self.material_dropdown.currentIndex())

    def _on_material_changed(self, index):
        """
        Called when the user selects a material.
        Updates density field and recalculates mass.
        """
        if index <= 0:
            self.density_input.setText("—")
            self._refresh_mass()
            return

        mat_key = self.material_dropdown.currentData()
        mat_data = self.materials.get(mat_key, {})
        density = mat_data.get("density [kg/mm^3]")

        if density is not None:
            self.density_input.setText(str(density))
        else:
            self.density_input.setText("N/A")

        self._refresh_mass()

    def _refresh_mass(self):
        """Recalculate and display mass based on current volume and density."""
        try:
            vol = self.node_data.get("volume [mm^3]")
            den_text = self.density_input.text().strip()
            den = float(den_text) if (den_text and den_text not in ("—", "N/A")) else None

            if vol is not None and den is not None:
                self.mass_label.setText(f"{round(vol * den, 6)} kg")
                self.mass_label.setStyleSheet("color: black;")
            else:
                self.mass_label.setText("—")
                self.mass_label.setStyleSheet("")
        except ValueError:
            self.mass_label.setText("Invalid density value")
            self.mass_label.setStyleSheet("color: red;")

    def get_data(self) -> dict:
        den_text = self.density_input.text().strip()
        density  = float(den_text) if (den_text and den_text not in ("—", "N/A")) else None
        vol      = self.node_data.get("volume [mm^3]")
        mass     = (vol * density) if (vol is not None and density is not None) else None
        mat_key  = self.material_dropdown.currentData()
        name     = self.name_input.text().strip() or self.node_data.get("name", "")

        return {
            "name":              name,
            "density [kg/mm^3]": density,
            "material":          mat_key,  # key from JSON
            "mass [kg]":         mass,
            "total mass [kg]":   mass,
        }

class SelectionDialog(QDialog):
    """
    A dialog for selecting a component for analysis.

    This class provides a user interface for selecting a specific component from the
    hierarchical structure of components for further analysis. It initializes the
    dialog with the provided project data and configuration settings.

    Attributes:
        structure (dict): The hierarchical structure of components loaded from the project.
        structure_file_path (str): The file path to the structure JSON file.
        selected_item (str or None): The name of the selected component.
        selected_data (dict or None): The data of the selected component.
        joining_connections_json (str): Path to the JSON file for joining connections.
        classification_json (str): Path to the JSON file for classification validation.

    Methods:
        __init__(data, file, config): Initializes the dialog with the given data, file, and configuration.
        init_ui(): Sets up the user interface for the dialog.
    """
    def __init__(self, data, file,config: Config):
        """
        Initialize the SelectionDialog.

        This constructor sets up the dialog with the provided hierarchical structure,
        file path, and configuration settings. It also initializes the user interface.

        Args:
            data (dict): The hierarchical structure of components.
            file (str): The file path to the structure JSON file.
            config (Config): An instance of the Config class containing configuration settings.

        Returns:
            None
        """
        super().__init__()
        self.setWindowTitle("Select Component for Analysis")
        self.structure = data
        self.structure_file_path = file
        self.selected_item = None
        self.selected_data = None
        self.init_ui()
        self.joining_connections_json = config.joining_connections_json
        self.classification_json = config.classification_json
    
    def init_ui(self):
        """
        Initialize the user interface for the SelectionDialog.

        This method sets up the UI components for the dialog, including a tree widget
        to display the hierarchical structure of components, and OK/Cancel buttons
        for user interaction. It also connects signals for selection changes and
        double-click events.

        Args:
            None

        Returns:
            None

        Notes:
            - The tree widget displays the hierarchical structure of components.
            - The OK button confirms the selection, while the Cancel button closes the dialog.
            - Signals are connected to handle selection changes and double-click events.
            - The window size is adjusted dynamically based on the tree size.
        """
        layout = QVBoxLayout(self)
        
        # Create tree widget to display the structure
        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["Component"])
        self.populate_tree(None, self.structure.get("data", {}))
        layout.addWidget(self.tree)
        
        # Add OK and Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.on_ok_clicked)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Connect selection change and double-click signals
        self.tree.itemSelectionChanged.connect(self.on_selection_changed)
        self.tree.itemDoubleClicked.connect(self.on_item_double_clicked)
        
        # Adjust the window size based on the tree size
        self.adjust_window_size()

    def adjust_window_size(self):
        """
        Adjust the window size based on the expanded tree size.

        This method calculates the required width and height of the dialog based on the
        size of the tree widget and its expanded items. It ensures that the dialog is
        resized to fit the content of the tree widget, with additional padding for better
        appearance.

        Args:
            None

        Returns:
            None

        Notes:
            - The tree is fully expanded before calculating the size.
            - Padding is added to both the width and height to ensure proper spacing.
        """
        self.tree.expandAll()
        width = self.tree.sizeHintForColumn(0) + 50  # Add some padding
        
        # Calculate the total height of all items in the tree
        def calculate_tree_height(item):
            height = self.tree.sizeHintForRow(0)
            for i in range(item.childCount()):
                height += calculate_tree_height(item.child(i))
            return height
        
        root = self.tree.invisibleRootItem()
        height = sum(calculate_tree_height(root.child(i)) for i in range(root.childCount())) + 100  # Add some padding
        
        self.resize(width, height)
    
    def populate_tree(self, parent_item, data):
        """
        Populate the tree widget with hierarchical component data.

        This method recursively populates the tree widget with the provided hierarchical
        component data. Each component is added as a `QTreeWidgetItem`, and its subcomponents
        are recursively added as child items.

        Args:
            parent_item (QTreeWidgetItem or None): The parent item in the tree to which the
                                                current component will be added. If `None`,
                                                the component is added as a top-level item.
            data (list): A list of dictionaries representing the hierarchical component data.

        Returns:
            None

        Notes:
            - Each component's data is stored in the `Qt.UserRole` of the corresponding tree item.
            - If a component has subcomponents, they are recursively added as child items.
            - The tree is fully expanded after all components are added.
        """
        for component_data in data:
            component_name = component_data.get("name", "Unnamed Component")
        
            if parent_item is None:
                tree_item = QTreeWidgetItem(self.tree, [component_name])
                self.tree.addTopLevelItem(tree_item)
            else:
                tree_item = QTreeWidgetItem(parent_item, [component_name])
                parent_item.addChild(tree_item)
        
            # Store the entire data in the item
            tree_item.setData(0, Qt.UserRole, component_data)
        
            # Recursively add subcomponents
            for subcomponent in component_data.get("components", []):
                subcomponent_name = subcomponent.get("name", "Unnamed Component")
                sub_item = QTreeWidgetItem(tree_item, [subcomponent_name])
                tree_item.addChild(sub_item)
                sub_item.setData(0, Qt.UserRole, subcomponent)
            
                if subcomponent.get("components"):
                    self.populate_tree(sub_item, subcomponent.get("components"))
            
            self.tree.expandAll()
    
    def get_unassessed_components(self):
        """
        Collect components that are not assessed.

        This method traverses the tree widget and collects the names of components
        that have not been assessed. A component is considered unassessed if it is
        a leaf node (has no children) and its `assessed` attribute is set to `False`
        or is missing.

        Args:
            None

        Returns:
            list: A list of names of unassessed components.

        Notes:
            - The method uses a recursive helper function to traverse the tree.
            - The `Qt.UserRole` is used to retrieve the component data stored in each tree item.
            - If a component does not have a name, it defaults to "Unnamed Component".
        """
        unassessed = []

        def recurse(item):
            component_data = item.data(0, Qt.UserRole)
            if component_data:
                if item.childCount() == 0 and not component_data.get('assessed', False):
                    unassessed.append(component_data.get("name", "Unnamed Component"))

            # Iterate over the children of the QTreeWidgetItem
            for i in range(item.childCount()):
                child_item = item.child(i)
                recurse(child_item)  # Recursively check child items

        root = self.tree.invisibleRootItem()
        recurse(root)
        return unassessed
    
    def get_top_level_assembly_id(self):
        """
        Get the ID of the top-level assembly from the structure data.

        This method retrieves the unique identifier (`id`) of the top-level assembly
        from the hierarchical structure data.

        Args:
            None

        Returns:
            str or None: The ID of the top-level assembly if it exists, otherwise `None`.

        Notes:
            - The top-level assembly is assumed to be the first item in the `data` list
            of the structure.
            - If the `data` key is missing or empty, the method returns `None`.
        """
        if self.structure.get("data"):
            return self.structure["data"][0].get("id")
        return None
    
    def is_top_level_assembly(self, item_data):
        """
        Check if the given item is the top-level assembly.

        This method determines whether the provided item's data corresponds to the
        top-level assembly in the hierarchical structure.

        Args:
            item_data (dict): The data of the item to check, including its `id`.

        Returns:
            bool: `True` if the item is the top-level assembly, `False` otherwise.

        Notes:
            - The top-level assembly ID is retrieved using the `get_top_level_assembly_id` method.
            - If the `id` of the given item matches the top-level assembly ID, the method returns `True`.
        """
        top_level_id = self.get_top_level_assembly_id()
        return item_data.get("id") == top_level_id
        
    
    def on_selection_changed(self):
        """
        Handle the selection change event in the tree widget.

        This method is triggered when the user selects a different item in the tree widget.
        It updates the `selected_item` attribute with the name of the selected component
        and the `selected_data` attribute with the corresponding data stored in the tree item.

        Args:
            None

        Returns:
            None

        Notes:
            - If no item is selected, the `selected_item` and `selected_data` attributes remain unchanged.
            - The `Qt.UserRole` is used to retrieve the data associated with the selected tree item.
        """
        selected_items = self.tree.selectedItems()
        if selected_items:
            self.selected_item = selected_items[0].text(0)
            self.selected_data = selected_items[0].data(0, Qt.UserRole)
    
    def on_item_double_clicked(self, item, column):
        """
        Handle the double-click event on a tree widget item.

        This method is triggered when the user double-clicks an item in the tree widget.
        If the item is not the top-level assembly and has no child components, it opens
        an `InfoDialog` to display and edit the item's data. If the dialog is accepted,
        the structure data is updated.

        Args:
            item (QTreeWidgetItem): The tree widget item that was double-clicked.
            column (int): The column index of the item that was double-clicked.

        Returns:
            None

        Notes:
            - The method does not open the `InfoDialog` for the top-level assembly or items
            with child components.
            - The `InfoDialog` allows the user to view and edit the selected item's data.
            - If the `InfoDialog` is closed with the OK button, the structure data is updated.
        """
        item_data = item.data(0, Qt.UserRole)
        if item_data:
            if self.is_top_level_assembly(item_data) or item.childCount() > 0:
                return  # Do not open the InfoDialog for the top-level assembly

            # Open InfoDialog with the selected item's data
            info_dialog = InfoDialog(item_data, self.structure, self.structure_file_path, Config(), self)
            if info_dialog.exec_() == QDialog.Accepted:
                # Load the updated data if the InfoDialog was closed with OK
                self.update_structure_data()
    
    def update_structure_data(self):
        """
        Load the current data from the JSON file and update the tree widget.

        This method reloads the hierarchical structure data from the JSON file and updates
        the tree widget to reflect the latest data.

        Args:
            None

        Returns:
            None

        Notes:
            - If the JSON file is successfully loaded, the `structure` attribute is updated.
            - The tree widget is cleared and repopulated with the updated structure data.
            - If the JSON file cannot be loaded, the method does nothing.
        """
        updated_data = load_json_file(self.structure_file_path)
        if updated_data:
            self.structure = updated_data  # Update the structure data

            self.tree.clear()  # Clear the tree
            self.populate_tree(None, self.structure.get("data", {}))  # Update the tree
    
    def on_ok_clicked(self):
        """
        Handle the OK button click event.

        This method is triggered when the user clicks the OK button in the dialog. It checks
        if all components in the structure are assessed. If there are unassessed components,
        it displays a warning message and prevents further action. If all components are
        assessed, it updates the assembly classifications and saves the updated structure
        back to the JSON file.

        Args:
            None

        Returns:
            None

        Notes:
            - If unassessed components are found, the method displays a warning and does not proceed.
            - The `update_assembly_classification` method is called to update classifications.
            - The updated structure is saved back to the JSON file using the `dump_file` function.
            - If all checks pass, the dialog is closed with `accept()`.
        """
        # Collect unassessed components
        unassessed_components = self.get_unassessed_components()
        
        if unassessed_components:
            # Create a warning message
            unassessed_list = "\n".join(unassessed_components)
            response = QMessageBox.warning(self, "Warning", f"The following components are not assessed:\n{unassessed_list}\n\nPlease complete the assessment of all components!")
            return
        else: 
            # If all components are assessed, update assembly classifications
            classification_structure = load_json_file(self.classification_json)
            data = self.structure
            self.update_assembly_classification(data, classification_structure)
            dump_file(data, self.structure_file_path)  # Save the updated structure back to the file

        self.accept()  # Proceed with the analysis

    def update_assembly_classification(self, data, classification_structure):
        """Derive assembly classifications from the leaf parts' classification paths."""
        return update_assembly_classifications(data, classification_structure)

    def collect_classifications(self, assembly):
        """Collect leaf classifications without using stale subassembly classifications."""
        return collect_part_classifications(assembly)

    def find_parent_classification_from_list(self, classification_structure, classifications):
        """Find the nearest shared parent across all depths of the classification tree."""
        return find_parent_classification(classification_structure, classifications)

class InfoDialog(QDialog):
    """
    A dialog for displaying and editing detailed information about a component.

    This class provides a user interface for viewing and modifying the details of a
    specific component, such as its name, material, classification, and other attributes.
    It also allows saving the updated information back to the project structure.

    Attributes:
        item_data (dict): The data of the selected component.
        data (dict): The hierarchical structure data of the project.
        structure_file_path (str): The file path to the structure JSON file.
        material_file_path (str): The file path to the materials JSON file.
        joining_json_path (str): The file path to the joining connections JSON file.
        classification_json (str): The file path to the classification JSON file.
        category_path (str): The file path to the category JSON file.

    Methods:
        __init__(item_data, data, file, config, parent): Initializes the dialog with the given data and configuration.
        init_ui(): Sets up the user interface for the dialog.
    """
    def __init__(self, item_data, data, file, config: Config, parent=None):
        """
        Initialize the InfoDialog.

        This constructor sets up the dialog with the provided component data, project structure,
        file paths, and configuration settings. It also initializes the user interface.

        Args:
            item_data (dict): The data of the selected component.
            data (dict): The hierarchical structure data of the project.
            file (str): The file path to the structure JSON file.
            config (Config): An instance of the Config class containing configuration settings.
            parent (QWidget, optional): The parent widget of the dialog. Defaults to None.

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle(f"Details of {item_data.get('name', 'Element')}")
        self.item_data = item_data
        self.data = data
        self.structure_file_path = file
        self.material_file_path = config.materials_json
        self.joining_json_path = config.joining_connections_json
        self.classification_json = config.classification_json
        self.category_path = config.category_file
        self.material_parameters = {}
        self.init_ui()
        self.deleted_connection = False
    
    def init_ui(self):
        """
        Initialize the user interface for the InfoDialog.

        This method sets up the UI components for the dialog, including input fields, checkboxes,
        dropdowns, and tree widgets for displaying and editing detailed information about a component.
        It dynamically handles categories and parameters based on the provided configuration.

        Args:
            None

        Returns:
            None

        Notes:
            - Categories such as "Basic Parameters," "Disassembly," "Material," and "Development" are dynamically handled.
            - Includes features like search bars, tree widgets, and buttons for user interaction.
            - The layout is scrollable to accommodate large amounts of data.
            - The "Save" button saves the updated data, and the "Assessed" checkbox tracks the assessment status.
        """
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_content.setLayout(form_layout)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        # Dictionary to hold parameter-widget pairs
        self.param_widgets = {}
        
        #Get data for the parameters
        category_data = load_json_file(self.category_path)
        categories = category_data.get('categories', {})
        self.param_mapping = category_data.get('param_mapping', {})

        # Get the material fractions, number of types and the used materials
        material_fractions, num_types, self.used_materials = self.calculate_material_fractions_components(self.item_data)

        # Iterate over categories and parameters
        for category, parameters in categories.items():
            group_box = QGroupBox(category)
            group_box.setObjectName(category)
            group_layout = QFormLayout()
            

            if category == "Basic Parameters":    
                
                self.basic_param_widgets = {}  # Initialize the dictionary to store basic parameters
                self.basic_parameters = {}  # Dictionary to store data for "Basic Parameters"

                for param in parameters:
                    key = self.param_mapping.get(param, param.lower().replace(" ", "_"))
                    value = self.item_data.get(key, "")

                    if param == "Hierarchy":
                        # Hierarchy-Pfad anzeigen
                        path = self.get_structure_path()
                        label_key = QLabel("<b>Hierarchy:</b>")
                        label_value = QLabel(path)
                        self.basic_param_widgets[key] = label_value  # Store the QLabel in basic_param_widgets
                        group_layout.addRow(label_key, label_value)
                        continue

                    elif param == "Classification":

                        # Classification
                        classification_group = QGroupBox("Classification")
                        self.classification_layout = QVBoxLayout()

                        # Create search bar for Classification
                        self.classification_search_bar = QLineEdit(self)
                        self.classification_search_bar.setPlaceholderText("Search classifications...")
                        self.classification_search_bar.textChanged.connect(self.filter_items)
    
                        self.classification_layout.addWidget(self.classification_search_bar)

                        # Create QTreeWidget for Classification
                        self.classification_tree = QTreeWidget()
                        self.classification_tree.setHeaderLabels(["Classification"])
                        self.populate_classification_tree()
                        self.classification_tree.expandAll()

                        self.classification_layout.addWidget(self.classification_tree)
                        classification_group.setLayout(self.classification_layout)
                        self.adjust_groupbox_size(classification_group)
                        group_layout.addRow(classification_group)
                        continue

                    elif param == "Standard Component": 
                        combo_box = QComboBox()
                        combo_box.addItems(["Yes", "No"])
                        key = self.param_mapping.get(param, param.lower().replace(" ", "_"))
                        current_value = self.item_data.get("basic_parameters", {}).get("standard_component", "No")  # Default to 'No' if not set
                        combo_box.setCurrentText(current_value)
                        group_layout.addRow(QLabel(f"<b>{param}:</b>"), combo_box)
                        self.basic_param_widgets[key] = combo_box  # Store the QComboBox in basic_param_widgets
                    
                    elif param == "Lifespan [a]":
                        line_edit = QLineEdit()
                        line_edit.setPlaceholderText(f"enter {param}")
                        # Access the "lifespan [a]" value from "basic_parameters" in self.item_data
                        key = self.param_mapping.get(param, param.lower().replace(" ", "_"))
                        current_value = self.item_data.get("basic_parameters", {}).get("lifespan [a]", None)  # Default to an empty string if not set

                        # Set the value of the QLineEdit if there is a current value
                        line_edit.setText(str(current_value))
                        group_layout.addRow(QLabel(f"<b>{param}:</b>"), line_edit)
                        # Add to param_widgets mapping
                        self.basic_param_widgets[key] = line_edit  # Store the QLineEdit in basic_param_widgets

                    else:
                        # Other parameters
                        if value:
                            label_key = QLabel(f"<b>{param}:</b>")
                            label_value = QLabel(str(value))
                            self.basic_param_widgets[key] = label_value  # Store the QLabel in basic_param_widgets
                            group_layout.addRow(label_key, label_value)
                        else:
                            line_edit = QLineEdit()
                            line_edit.setPlaceholderText(f"enter {param}")
                            group_layout.addRow(QLabel(f"<b>{param}:</b>"), line_edit)
                            # Add to param_widgets mapping
                            self.basic_param_widgets[key] = line_edit  # Store the QLineEdit in basic_param_widgets

            elif category == "Disassembly":

                self.disassembly_parameters = {}  # Dictionary to store all disassembly-related parameters

                #Check if item is a joining component
                self.is_joining_component_checkbox = QCheckBox(f"Is {self.item_data.get('name','component')} a Joining Component?")
                joining_component = self.item_data.get("disassembly_parameters", {}).get("is_joining_component", False)
                self.is_joining_component_checkbox.setChecked(joining_component)
                group_layout.addRow(self.is_joining_component_checkbox)

                self.joining_component_group = QGroupBox("Which Components are Joined?")
                self.joining_component_layout = QVBoxLayout()

                # Create a search bar for joining components
                self.joining_component_search_bar = QLineEdit()
                self.joining_component_search_bar.setPlaceholderText("Search components...")
                self.joining_component_search_bar.textChanged.connect(self.filter_items)
                self.joining_component_layout.addWidget(self.joining_component_search_bar)

                # Create the tree for joining components
                self.joining_component_tree = QTreeWidget()
                self.joining_component_tree.setHeaderLabels(["Component"])
                self.joining_component_layout.addWidget(self.joining_component_tree)

                # Populate the tree with components
                self.populate_disassembly_tree(self.joining_component_tree, self.data.get('data', []), "joining_component")

                # Connect the checkbox to toggle the tree and search bar visibility
                self.is_joining_component_checkbox.stateChanged.connect(self.toggle_joining_component_group)

                # Expand all items in the tree
                self.joining_component_tree.expandAll()

                # Add the joining component group to the layout
                self.joining_component_group.setLayout(self.joining_component_layout)
                self.adjust_groupbox_size(self.joining_component_group)
                group_layout.addRow(self.joining_component_group)

                # Connected With
                self.connected_with_group = QGroupBox("Connected With")
                self.connected_with_layout = QVBoxLayout()

                # Search bar for covered_by
                self.connected_with_search_bar = QLineEdit()
                self.connected_with_search_bar.setPlaceholderText("Search components...")
                self.connected_with_search_bar.textChanged.connect(self.filter_items)  # Connect to filtering method
                self.connected_with_layout.addWidget(self.connected_with_search_bar)

                # Create the tree widget for connected with
                self.connected_with_tree = QTreeWidget()
                self.connected_with_tree.setHeaderLabels(["Component"])
                self.populate_disassembly_tree(self.connected_with_tree, self.data.get('data', []), 'connected_with')

                # Connect the itemChanged signal to dynamically update the connected_by list
                self.connected_with_tree.itemChanged.connect(self.update_connected_by_list)

                # Expand all items in the tree
                self.connected_with_tree.expandAll()

                # Add the tree to the layout
                self.connected_with_layout.addWidget(self.connected_with_tree)
                self.connected_with_group.setLayout(self.connected_with_layout)
                self.adjust_groupbox_size(self.connected_with_group)
                group_layout.addRow(self.connected_with_group)

                # Connected With
                self.connected_by_group = QGroupBox("Connected By")
                self.connected_by_layout = QVBoxLayout()

                self.assigned_connections = {} # Dictionary to track assigned connections for each component

                self.connected_by_list = QListWidget()
                self.connected_by_layout.addWidget(self.connected_by_list)
                # Call update_connected_by_list to handle preset checked items
                self.update_connected_by_list()

                # Search bar for joining connections
                self.joining_search_bar = QLineEdit()
                self.joining_search_bar.setPlaceholderText("Search joining connections...")
                self.joining_search_bar.textChanged.connect(self.filter_items)  # Connect to filtering method
                self.connected_by_layout.addWidget(self.joining_search_bar)

                #Tree Widget for Joining Connections
                self.joining_tree = QTreeWidget()
                self.joining_tree.setHeaderLabels(["Connection", "Context/Example"])
                self.joining_tree.setAlternatingRowColors(True)  # Enable alternating row colors
                self.load_joining_connections()

                # Expand all items in the tree
                self.joining_tree.expandAll()
                self.connected_by_layout.addWidget(self.joining_tree)  # Add tree to the layout

                add_and_delete_buttons_layout = QHBoxLayout()

                # Add a button to assign joining connections
                self.add_connection_button = QPushButton("Add Connection")
                self.add_connection_button.clicked.connect(self.add_connection_to_selected_entry)
                add_and_delete_buttons_layout.addWidget(self.add_connection_button)

                # Add a button to assign joining connections
                self.add_deletion_button = QPushButton("Delete connection(s)")
                self.add_deletion_button.clicked.connect(self.delete_connections_to_selected_entry)
                add_and_delete_buttons_layout.addWidget(self.add_deletion_button)

                self.connected_by_layout.addLayout(add_and_delete_buttons_layout)

                self.connected_by_group.setLayout(self.connected_by_layout)
                self.adjust_groupbox_size(self.connected_by_group)
                group_layout.addRow(self.connected_by_group)
            
                # Covered by
                self.covered_by_group = QGroupBox("Covered by")
                self.covered_by_layout = QVBoxLayout()

                # Search bar for covered_by
                self.covered_by_search_bar = QLineEdit()
                self.covered_by_search_bar.setPlaceholderText("Search components...")
                self.covered_by_search_bar.textChanged.connect(self.filter_items)  # Connect to filtering method
                self.covered_by_layout.addWidget(self.covered_by_search_bar)

                # Create the tree widget for covered by
                self.covered_by_tree = QTreeWidget()
                self.covered_by_tree.setHeaderLabels(["Component"])
                self.populate_disassembly_tree(self.covered_by_tree, self.data.get('data', []), 'covered_by')

                # Expand all items in the tree
                self.covered_by_tree.expandAll()
        
                self.covered_by_layout.addWidget(self.covered_by_tree)
                self.covered_by_group.setLayout(self.covered_by_layout)
                self.adjust_groupbox_size(self.covered_by_group)
                group_layout.addRow(self.covered_by_group)

                # Initialize the tree and search bar visibility based on the checkbox state
                self.toggle_joining_component_group()
            
            elif category == "Material":

                # Load materials database
                self.materials_db = load_json_file(self.material_file_path)

                # Create layout
                materials_layout = QVBoxLayout()

                # Label
                materials_label = QLabel("<b>Material:</b>")
                materials_layout.addWidget(materials_label)

                # Container for material row
                self.material_row_container = QWidget()
                self.material_row_layout = QHBoxLayout()
                self.material_row_container.setLayout(self.material_row_layout)
                materials_layout.addWidget(self.material_row_container)

                # Get current material (from item_data)
                current_material = self.used_materials[0] if self.used_materials else "(none)"

                # Dropdown: material type (editable)
                self.material_combo = QComboBox()
                self.material_combo.addItem("Select material…", None)
                for name in sorted(self.materials_db.get("materials", {}).keys()):
                    self.material_combo.addItem(name, name)
                # Pre-select current material if in DB
                idx = self.material_combo.findData(current_material)
                if idx >= 0:
                    self.material_combo.setCurrentIndex(idx)
                else:
                    self.material_combo.addItem(current_material, current_material)
                    self.material_combo.setCurrentIndex(self.material_combo.count() - 1)

                # Button: assess material
                self.assess_material_button = QPushButton("Assess Material")
                is_assessed = current_material in getattr(self, "material_parameters", {})
                self.assess_material_button.setEnabled(not is_assessed)
                self.assess_material_button.clicked.connect(self._on_assess_material_button_clicked)

                # Connect combo change to status update
                self.material_combo.currentIndexChanged.connect(self._update_material_status)

                self.material_row_layout.addWidget(QLabel("Material:"))
                self.material_row_layout.addWidget(self.material_combo, 2)
                self.material_row_layout.addWidget(self.assess_material_button)

                # Count label
                self.materials_count_label = QLabel(f"Number of Materials: {len(self.used_materials)}")
                materials_layout.addWidget(self.materials_count_label)


                group_layout.addRow(materials_layout)

            elif category == "Development":

                self.development_param_widgets = {}  # Dictionary to store all development-related parameters
                self.development_parameters = {}  # Dictionary to store data for "Development"

                # Get development parameters from self.item_data
                development_params = self.item_data.get("development_parameters", {})

                # Dynamically handle Development parameters from JSON
                for param in parameters:
                    combo_box = QComboBox()
                    combo_box.addItems(["Yes", "No"])
                    key = self.param_mapping.get(param, param.lower().replace(" ", "_"))

                    # Get the current value from development_parameters if it exists, otherwise default to 'No'
                    current_value = development_params.get(key, 'No')
                    combo_box.setCurrentText(current_value)

                    # Store the comboBox in the dictionary for later access
                    self.development_param_widgets[key] = combo_box

                    # Add the comboBox to the layout
                    group_layout.addRow(QLabel(f"<b>{param}:</b>"), combo_box)

            
            group_box.setLayout(group_layout)
            form_layout.addRow(group_box)


        # Add Assessed Checkbox
        self.assessed_checkbox = QCheckBox("Assessed")
        self.assessed_checkbox.setChecked(self.item_data.get('assessed', False))  # Set checked state based on existing data
        form_layout.addRow(self.assessed_checkbox)
        
        # Save and Close buttons 
        button_layout = QVBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_data)
        button_layout.addWidget(save_button)
        
        layout.addLayout(button_layout)

        self.setLayout(layout)  #Set the layout
        self.adjustSize() 

    def adjust_groupbox_size(self, group_box):
        """
        Adjust the size of the group box based on its content.

        This method dynamically adjusts the minimum height of the provided group box
        to ensure it accommodates its content with additional padding for better appearance.

        Args:
            group_box (QGroupBox): The group box whose size needs to be adjusted.

        Returns:
            None

        Notes:
            - The method uses the `sizeHint` of the group box to calculate the required height.
            - An additional padding of 30 pixels is added to the calculated height.
        """
        group_box.setMinimumHeight(group_box.sizeHint().height() + 30)  # 30 pixels für Padding
    
    def get_structure_path(self) -> str:
        """
        Determine the structural path of the current element.

        This method traverses the hierarchical structure to find the path of the current
        element based on its unique identifier (`id`). The path is returned as a string
        with components separated by " > ".

        Args:
            None

        Returns:
            str: The structural path of the current element. If the element is not found,
                the method returns "Pfad nicht gefunden" (Path not found).

        Notes:
            - The method uses a recursive helper function to traverse the hierarchy.
            - The `id` of the current element is retrieved from `self.item_data`.
            - The path is constructed by appending the names of components in the hierarchy.
        """
        target_id = self.item_data.get('id')
        data = self.data

        def recurse(items: List[Dict[str, Any]], path: List[str]) -> List[str]:
            for item in items:
                if item['id'] == target_id:
                    return path + [item['name']]
                if 'components' in item and item['components']:
                    result = recurse(item['components'], path + [item['name']])
                    if result:
                        return result
            return []

        path = recurse(data.get('data', []), [])
        return " > ".join(path) if path else "Pfad nicht gefunden"
    
    def populate_classification_tree(self):
        """
        Populate the classification tree with data from the classification JSON, handling 'Examples' appropriately.

        This method populates the classification tree widget with hierarchical data from the classification JSON file.
        It also presets the checkbox states based on the existing classification data in the `basic_parameters` of the
        current item.

        Args:
            None

        Returns:
            None

        Notes:
            - The method skips the "Examples" key in the classification data.
            - If a classification item matches an entry in the tree, its checkbox is preset to checked.
            - The tree is cleared before populating to avoid duplicate entries.
            - Nested structures in the classification data are handled recursively.
        """
        self.classification_data = load_json_file(self.classification_json)

        # Get the classification text that is already in basic_parameters
        classification_item = self.item_data.get("basic_parameters", {}).get("classification", [])
        if classification_item is None:
            classification_item = []  # Default to an empty list if the key is missing or None

        def populate_tree(parent_item, data):
            """
            Recursively populate the tree with any nested structure.

            Args:
                parent_item (QTreeWidgetItem): The parent tree item to which child items will be added.
                data (dict or list): The hierarchical classification data to populate the tree with.

            Returns:
                None
            """
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == "Examples":
                        continue  # Skip the 'Examples' key
                
                    # Check if the dictionary only contains the "Examples" key
                    if isinstance(value, dict) and len(value) == 1 and "Examples" in value:
                        tree_item = QTreeWidgetItem(parent_item, [key])
                        # Preset checkbox state based on classification_item
                        if key in classification_item:
                            tree_item.setCheckState(0, Qt.Checked)
                        else:
                            tree_item.setCheckState(0, Qt.Unchecked)
                    else:
                        tree_item = QTreeWidgetItem(parent_item, [key])
                        populate_tree(tree_item, value)  # Recursively populate children
            elif isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        # Add dictionaries as tree items
                        for subkey, subvalue in entry.items():
                            tree_item = QTreeWidgetItem(parent_item, [subkey])
                            tree_item.setCheckState(0, Qt.Unchecked)  # Add a checkbox for this item
                    else:
                        # Add non-dictionary entries as tree items
                        tree_item = QTreeWidgetItem(parent_item, [entry])
                        if entry in classification_item:
                            tree_item.setCheckState(0, Qt.Checked)  # Set checkbox to checked
                        else:
                            tree_item.setCheckState(0, Qt.Unchecked)  # Set checkbox to unchecked

        # Clear the tree before populating
        self.classification_tree.clear()
        populate_tree(self.classification_tree.invisibleRootItem(), self.classification_data)
    
    def toggle_joining_component_group(self):
        """
        Toggle the visibility of the joining, connected_with, and connected_by groups based on the checkbox state.

        This method adjusts the visibility of the `joining_component_group`, `connected_with_group`, 
        and `connected_by_group` depending on whether the "Is Joining Component" checkbox is checked.

        Args:
            None

        Returns:
            None

        Notes:
            - If the checkbox is checked, the `joining_component_group` is shown, and the 
            `connected_with_group` and `connected_by_group` are hidden.
            - If the checkbox is unchecked, the `connected_with_group` and `connected_by_group` 
            are shown, and the `joining_component_group` is hidden.
        """
        is_checked = self.is_joining_component_checkbox.isChecked()

        # Show or hide the joining_component_group
        self.joining_component_group.setVisible(is_checked)

        # Hide or show the connected_with_group and connected_by_group
        self.connected_with_group.setVisible(not is_checked)
        self.connected_by_group.setVisible(not is_checked)
    
    def populate_disassembly_tree(self, tree_widget, data, key):
        """
        Populate a tree widget with component names based on a given key.

        This method populates a tree widget with hierarchical component data, presetting
        checkboxes based on the provided key. It handles the main assembly and its subcomponents
        recursively.

        Args:
            tree_widget (QTreeWidget): The tree widget to populate.
            data (list): A list of dictionaries representing the hierarchical component data.
            key (str): The key used to determine which components should have their checkboxes preset.

        Returns:
            None

        Notes:
            - The `key` determines which data to use for presetting checkboxes:
                - 'connected_with': Uses `assigned_connections` from `disassembly_parameters`.
                - 'covered_by': Uses `covered_by` from `disassembly_parameters`.
                - 'joining_component': Uses `joining_components` from `disassembly_parameters`.
            - The main assembly is added as the root item without a checkbox.
            - Subcomponents are added recursively, with checkboxes preset based on the `key`.
            - Components with no subcomponents are treated as leaf nodes and have checkboxes.
        """
        # Determine which data to use for presetting checkboxes based on the key
        if key == 'connected_with':
            checked_components = set(self.item_data.get("disassembly_parameters", {}).get("assigned_connections", {}).keys())
        elif key == 'covered_by':
            checked_components = set(self.item_data.get("disassembly_parameters", {}).get("covered_by", []))
        elif key == 'joining_component':
            checked_components = set(self.item_data.get("disassembly_parameters", {}).get("joining_components", []))
        else:
            checked_components = set()  # Default to an empty set if the key is not recognized

        if data:
            # Create a tree item for the main assembly without a checkbox
            main_assembly_item = QTreeWidgetItem(tree_widget, [data[0].get("name", "Unnamed Assembly")])
            main_assembly_item.setData(0, Qt.UserRole, data[0])

            # Add subcomponents for the main assembly
            self.add_subcomponents(main_assembly_item, data[0].get("components", []), checked_components)

        for component_data in data[1:]:
            component_id = component_data.get("name")

            if component_id != self.item_data.get("name"):  # Exclude the current item
                component_name = component_data.get("name", "Unnamed Component")

                # Create a tree item for this component
                tree_item = QTreeWidgetItem(tree_widget, [component_name])
                tree_item.setData(0, Qt.UserRole, component_data)

                # Only add a checkbox if the component has no subcomponents
                if not component_data.get("components"):
                    if component_id in checked_components:
                        tree_item.setCheckState(0, Qt.Checked)  # Set checkbox to checked
                    else:
                        tree_item.setCheckState(0, Qt.Unchecked)  # Set checkbox to unchecked

                self.add_subcomponents(tree_item, component_data.get("components", []), checked_components)
        
    def add_subcomponents(self, parent_item, subcomponents, checked_components):
        """
        Recursively adds subcomponents to the tree.

        This method adds subcomponents to a parent tree item, recursively handling nested
        subcomponents. It also presets checkboxes for leaf nodes based on the provided
        `checked_components`.

        Args:
            parent_item (QTreeWidgetItem): The parent tree item to which subcomponents will be added.
            subcomponents (list): A list of dictionaries representing the subcomponents.
            checked_components (set): A set of component names that should have their checkboxes preset.

        Returns:
            None

        Notes:
            - Subcomponents with no further children are treated as leaf nodes and have checkboxes.
            - The method excludes the current item from being added as a subcomponent.
            - The `Qt.UserRole` is used to store the subcomponent data in the tree item.
        """
        for subcomponent in subcomponents:
            subcomponent_id = subcomponent.get("name")

            if subcomponent_id != self.item_data.get("name"):  # Exclude the current item
                sub_item = QTreeWidgetItem(parent_item, [subcomponent.get("name", "Unnamed Component")])
                sub_item.setData(0, Qt.UserRole, subcomponent)

                # Only add a checkbox if the subcomponent has no further subcomponents
                if not subcomponent.get("components"):
                    if subcomponent_id in checked_components:
                        sub_item.setCheckState(0, Qt.Checked)  # Set checkbox to checked
                    else:
                        sub_item.setCheckState(0, Qt.Unchecked)  # Set checkbox to unchecked

                # Recursively add children
                if subcomponent.get("components"):
                    self.add_subcomponents(sub_item, subcomponent.get("components", []), checked_components)

    def update_connected_by_list(self):
        """
        Updates the list of checked components in the connected_by group box.

        This method collects all checked components from the `connected_with_tree` widget
        and updates the `connected_by_list` widget to display the selected components along
        with their assigned connections. It also synchronizes the `self.assigned_connections`
        attribute with the `assigned_connections` data from `item_data`.

        Args:
            None

        Returns:
            None

        Notes:
            - Checked components are collected recursively from the `connected_with_tree`.
            - If a component has assigned connections, the connection details are included
            in the `connected_by_list` entry.
            - Connection details include attributes such as amount, non-destructive status,
            ageing, tool, and detachment time.
        """
        checked_components = []

        def collect_checked_items(item):
            """Recursively collect checked items."""
            if item.checkState(0) == Qt.Checked:
                checked_components.append(item.text(0))  # Add the component name to the list
            for i in range(item.childCount()):
                collect_checked_items(item.child(i))
        
        # Start from the root items of the tree
        root = self.connected_with_tree.invisibleRootItem()
        for i in range(root.childCount()):
            collect_checked_items(root.child(i))
            
        # Get the assigned connections from item_data
        assigned_connections = self.item_data.get("disassembly_parameters", {}).get("assigned_connections", {})

        # Synchronize self.assigned_connections with assigned_connections
        self.assigned_connections = assigned_connections

        # Update the connected_by_list while preserving assigned connections
        self.connected_by_list.clear()
        for component in checked_components:
            if component in assigned_connections:
                # Include assigned connections in the list entry
                connections = ", ".join(
                [
                    f"{conn} (Amount: {data['amount']}, Non-Destructive: {'Yes' if data['non_destructive'] else 'No'}, "
                    f"Ageing: {'Yes' if data['ageing'] else 'No'}, Tool: {data['tool'] or 'None'}, "
                    f"Detachment Time: {data['detachment_time']}s)"
                    for conn, data in assigned_connections[component].items()
                ]
            )
                self.connected_by_list.addItem(f"{component} | Connections: {connections}")
            else:
                # Add the component without connections
                self.connected_by_list.addItem(component)
    
    def load_joining_connections(self):
        """
        Load joining connections from the JSON file and populate the tree.

        This method reads joining connection data from a JSON file and populates the
        `joining_tree` widget with the connections. It also checks if there are any
        joining connections in the `item_data` and updates the tree accordingly.

        Args:
            None

        Returns:
            None

        Notes:
            - The JSON file is loaded from the path specified in `self.joining_json_path`.
            - Connections are added to the tree using the `add_connections_to_tree` method.
            - If a connection in `item_data` is not found in the tree, a warning message is displayed.
        """
        with open(self.joining_json_path, 'r', encoding='utf-8') as f:
            connections = json.load(f)
        self.add_connections_to_tree(connections, None)

        # Check if there are joining connections in item_data
        if 'joining_connections' in self.item_data:
            for conn in self.item_data['joining_connections']:
                connection_name = conn['type']
                found = False
                for i in range(self.joining_tree.topLevelItemCount()):
                    item = self.joining_tree.topLevelItem(i)
                    if self.set_spin_value(item, connection_name):
                        found = True
                        break  # Stop searching once we've set the value

                if not found:
                    QMessageBox.warning(self, "Warning", f"Connection '{connection_name}' not found in the tree.")

    def add_connections_to_tree(self, connections, parent_item):
        """
        Recursively adds connections to the QTreeWidget.

        This method populates the `joining_tree` widget with hierarchical joining connection data.
        Each connection is added as a `QTreeWidgetItem`, and sub-connections are recursively added
        as child items. A spin box is added to leaf nodes for specifying connection quantities.

        Args:
            connections (list): A list of dictionaries representing the joining connections.
                                Each dictionary contains the keys 'name', 'context', and 'sub_connections'.
            parent_item (QTreeWidgetItem or None): The parent tree item to which connections will be added.
                                                If `None`, connections are added as top-level items.

        Returns:
            None

        Notes:
            - Each connection is displayed with its name and context in the tree widget.
            - Sub-connections are recursively added as child items.
            - A spin box is added to leaf nodes to allow users to specify the quantity of the connection.
        """
        for conn in connections:
            item = QTreeWidgetItem([conn['name'], conn['context']])  # Add context to the item
            if parent_item is None:
                self.joining_tree.addTopLevelItem(item)
            else:
                parent_item.addChild(item)

            if conn['sub_connections']:
                self.add_connections_to_tree(conn['sub_connections'], item)
            else:
                spin = QSpinBox()
                spin.setRange(0, 1000)
                spin.setValue(0)
                self.joining_tree.setItemWidget(item, 2, spin)
    
    def delete_connections_to_selected_entry(self):
        """
        Delete all joining connections of the selected entry in the connected_by_list.

        This method allows the user to delete all connections of a selected component
        in the `connected_by_list` and updates the `assigned_connections` dictionary and the list item text.

        Args:
            None

        Returns:
            None

        Notes:
            - If no entry is selected in the `connected_by_list`, a warning is displayed.
            - The `assigned_connections` dictionary is updated with the new connection data.
            - The text of the selected list item is updated to reflect the assigned connections.
        """
        # Get the selected entry in the connected_by_list
        selected_item = self.connected_by_list.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "Warning", "Please select an entry in the Connected By list.")
            return

        # Get the component name from the selected list item
        component_name = selected_item.text().split(" |")[0]

        # Update the assigned_connections dictionary
        # Purge all connections
        self.assigned_connections[component_name] = {}
        
        self.deleted_connection = True
         # Load the existing category data
        category_data = load_json_file(self.category_path)
        categories = category_data.get('categories', {})
        self.param_mapping = category_data.get('param_mapping', {})

        # Collect data for each category
        for category in categories.keys():
            self.collect_category_data(category)

        # Validate all parameters
        if not self.validate_all_parameters(categories):
            return  # Stop saving if validation fails

        self.update_json_data()  # Update the JSON data

        self.update_connected_by_list()
    
    def add_connection_to_selected_entry(self):
        """
        Assign a joining connection and amount to the selected entry in the connected_by_list.

        This method allows the user to assign a joining connection to a selected component
        in the `connected_by_list`. It collects additional information about the connection
        (e.g., amount, non-destructive status, ageing, tool, detachment time) using a custom
        dialog and updates the `assigned_connections` dictionary and the list item text.

        Args:
            None

        Returns:
            None

        Notes:
            - If no entry is selected in the `connected_by_list`, a warning is displayed.
            - If no joining connection is selected in the `joining_tree`, a warning is displayed.
            - The `JoiningConnectionDialog` is used to collect additional connection details.
            - The `assigned_connections` dictionary is updated with the new connection data.
            - The text of the selected list item is updated to reflect the assigned connections.
        """
        # Get the selected entry in the connected_by_list
        selected_item = self.connected_by_list.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "Warning", "Please select an entry in the Connected By list.")
            return

        # Get the selected joining connection from the joining_tree
        selected_connection_item = self.joining_tree.currentItem()
        if not selected_connection_item:
            QMessageBox.warning(self, "Warning", "Please select a joining connection from the tree.")
            return

        # Prompt the user to enter an amount for the connection
        connection_name = selected_connection_item.text(0)  # Get the connection name
        # Open the custom dialog to collect all required information
        dialog = JoiningConnectionDialog(connection_name, self)
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()  # Get the data entered by the user

            # Get the component name from the selected list item
            component_name = selected_item.text().split(" |")[0]

            # Update the assigned_connections dictionary
            if component_name not in self.assigned_connections:
                self.assigned_connections[component_name] = {}
            self.assigned_connections[component_name][connection_name] = data

            # Update the selected list item text
            connections = ", ".join(
                [
                    f"{conn} (Amount: {data['amount']}, Non-Destructive: {'Yes' if data['non_destructive'] else 'No'}, "
                    f"Ageing: {'Yes' if data['ageing'] else 'No'}, Tool: {data['tool'] or 'None'}, "
                    f"Detachment Time: {data['detachment_time']}s)"
                    for conn, data in self.assigned_connections[component_name].items()
                ]
            )
            selected_item.setText(f"{component_name} | Connections: {connections}")

    def get_checked_items_from_tree(self, tree_widget):
        """
        Collect the checked items from a tree widget.

        This method traverses the provided tree widget and collects the names of all items
        that are checked. It recursively processes all child items to ensure the entire tree
        is covered.

        Args:
            tree_widget (QTreeWidget): The tree widget to traverse.

        Returns:
            list: A list of names of the checked items.

        Notes:
            - The method uses a recursive helper function to traverse the tree.
            - Only the names of checked items are stored in the returned list.
        """
        checked_items = []

        def collect_checked_items(item):
            """Recursively collect checked items."""
            if item.checkState(0) == Qt.Checked:
                checked_items.append(item.text(0))  # Store only the component name
            for i in range(item.childCount()):
                collect_checked_items(item.child(i))

        # Start from the root items of the tree
        root = tree_widget.invisibleRootItem()
        for i in range(root.childCount()):
            collect_checked_items(root.child(i))
        
        return checked_items

    def calculate_material_fractions_components(self, item_data):
        """
        Calculate the mass fractions of materials and return the used materials.

        This method calculates the percentage of each material's mass relative to the total mass
        of the given component and its subcomponents. It also returns the number of unique materials
        and a list of the materials used.

        Args:
            item_data (dict): The data of the component, including its mass, material, and subcomponents.

        Returns:
            tuple: A tuple containing:
                - fractions (list): A list of dictionaries, each containing:
                    - 'type' (str): The material type.
                    - 'percentage' (float): The percentage of the material's mass relative to the total mass.
                - num_materials (int): The number of unique materials used.
                - materials_list (list): A list of material names.

        Notes:
            - The method uses a recursive helper function to traverse the component hierarchy.
            - If the total mass is zero, the percentage for each material is set to 0.
        """
        materials = defaultdict(float)
        total_mass = item_data.get("total mass [kg]", 0)

        def recurse(data):
            mass = data.get("mass [kg]", 0) or data.get("total mass [kg]", 0)
            material = data.get("material")
            if material:
                materials[material] += mass
            for component in data.get("components", []):
                recurse(component)

        recurse(item_data)

        fractions = []
        for mat, mass in materials.items():
            percentage = (mass / total_mass * 100) if total_mass > 0 else 0
            fractions.append({"type": mat, "percentage": round(percentage, 2)})
        
        return fractions, len(materials), list(materials.keys())  # Return materials as well
    
    def update_materials_database(self, material_name, material_data):
        """
        Updates the materials database with new material data.

        This method checks if the specified material already exists in the materials database.
        If it exists, the material's data is updated. If it does not exist, the material is added
        to the database. The updated database is then saved back to the JSON file.

        Args:
            material_name (str): The name of the material to update or add.
            material_data (dict): The data of the material to update or add.

        Returns:
            None

        Notes:
            - If the material already exists, its data is updated with the provided `material_data`.
            - If the material does not exist, it is added to the database.
            - The updated database is saved to the file specified by `self.material_file_path`.
        """
        # Check if the material already exists in the database
        if material_name in self.materials_db['materials']:
            # Update existing material
            self.materials_db['materials'][material_name].update(material_data)
            print(f"Updated existing material: {material_name}")
        else:
            # Add new material
            self.materials_db['materials'][material_name] = material_data
            print(f"Added new material: {material_name}")

        # Save the updated materials database back to the JSON file
        dump_file(self.materials_db, self.material_file_path)

    def enable_assess_button(self):
        """
        Enable the 'Assess Material' button when a material is selected.

        This method checks if any item in the `materials_list` is selected. If at least one
        item is selected, the 'Assess Material' button is enabled; otherwise, it is disabled.

        Args:
            None

        Returns:
            None
        """
        selected_items = self.materials_list.selectedItems()
        self.assess_material_button.setEnabled(bool(selected_items))

    def assess_selected_material(self):
        """
        Open the MaterialParameterDialog for the currently selected material in the dropdown.

        This method retrieves the currently selected material from the dropdown,
        opens the MaterialParameterDialog, and updates the button state.

        Args:
            None

        Returns:
            None
        """
                # Get selected material key
        mat_key = self.material_combo.currentData()
        if not mat_key:
            QMessageBox.warning(self, "No Material Selected", "Please select a material first.")
            return

        # Open the MaterialParameterDialog
        dialog = MaterialParameterAssessmentDialog(mat_key, self)
        if dialog.exec() == QDialog.Accepted:
            # Get the data entered by the user
            material_data = dialog.get_assessment_data()

            # Initialize self.material_parameters if not exists
            if not hasattr(self, "material_parameters"):
                self.material_parameters = {}

            # Store the material data
            self.material_parameters[mat_key] = material_data

            # Disable button (material now assessed)
            self.assess_material_button.setEnabled(False)

            # Optional: update dropdown text to show status
            # self.material_combo.setItemText(self.material_combo.currentIndex(), f"{mat_key} ✓")
    
    def open_material_parameter_dialog(self, item):
        """
        Open the MaterialParameterDialog for the selected material.

        This method opens the `MaterialParameterDialog` for the selected material, allowing the user
        to view or edit its parameters. The updated material data is saved to the materials database
        and reflected in the UI.

        Args:
            item (QListWidgetItem): The selected material item from the `materials_list`.

        Returns:
            None

        Notes:
            - The material name is extracted from the item's text.
            - If the dialog is accepted, the material data is updated in the database and the UI.
            - The item's text is updated to indicate that parameters have been set.
        """
        # Extract the material name from the clicked item
        material_name = item.text().split(" (")[0]  # Get the material type 

        # Open the MaterialParameterDialog
        dialog = MaterialParameterAssessmentDialog(material_name, self)
        if dialog.exec_() == QDialog.Accepted:
            # Get the data entered by the user
            material_data = dialog.get_assessment_data()
            # Store the collected data
            if not hasattr(self, "material_parameters"):
                self.material_parameters = {}
            self.material_parameters[material_name] = material_data

            # Update the materials database with the collected data
            self.update_materials_database(material_name, material_data)

            # Update the item text to indicate parameters have been set
            item.setText(f"{material_name} (Parameters Set)")

    def _on_assess_material_button_clicked(self):
        """Called when the 'Assess Material' button is clicked."""
        mat_key = self.material_combo.currentData()
        if not mat_key:
            QMessageBox.warning(self, "No Material Selected", "Please select a material first.")
            return
        self._assess_material_now(mat_key)

    def _assess_material_now(self, mat_key: str) -> bool:
        """Open the MaterialParameterDialog for the given material. Returns True if accepted."""
        dialog = MaterialParameterAssessmentDialog(mat_key, self)
        if dialog.exec() == QDialog.Accepted:
            # Get the data entered by the user
            material_data = dialog.get_assessment_data()

            # Store in self.material_parameters
            if not hasattr(self, "material_parameters"):
                self.material_parameters = {}
            self.material_parameters[mat_key] = material_data

            # Disable button (now assessed)
            self.assess_material_button.setEnabled(False)

            # Update dropdown text to show status
            self._update_material_status()
            return True
        return False

    def _update_material_status(self):
        """Update the dropdown text to show whether the material is assessed."""
        if not hasattr(self, "material_combo"):
            return
        mat_key = self.material_combo.currentData()
        if not mat_key:
            return

        # Check if fully assessed
        if mat_key in self.material_parameters:
            data = self.material_parameters[mat_key]
            required = [
                "recycling_percentage", "recyclate_content", "recycling_criticality",
                "incineration_for_energy_recovery",
                "environmental_harmfulness", "health_harmfulness", "monomaterial",
                "additives_or_fillers", "surface_coatings", "availability"
            ]
            if all(data.get(f) is not None for f in required):
                idx = self.material_combo.currentIndex()
                self.material_combo.setItemText(idx, f"{mat_key} ✓")
            else:
                idx = self.material_combo.currentIndex()
                self.material_combo.setItemText(idx, f"{mat_key} (partial)")
        else:
            idx = self.material_combo.currentIndex()
            self.material_combo.setItemText(idx, mat_key)


    def filter_items(self):
        """
        Filters both the covered by tree and the joining tree based on the search input.

        This method filters items in various tree widgets based on the text entered in the corresponding
        search bar. It hides items that do not match the search text and shows items that do.

        Args:
            None

        Returns:
            None

        Notes:
            - The method determines which tree to filter based on the sender of the signal.
            - The `filter_item` method is used to recursively filter items in the tree.
            - Supported trees include:
                - `covered_by_tree`
                - `joining_tree`
                - `classification_tree`
                - `connected_with_tree`
                - `joining_component_tree`
        """
        sender = self.sender()
        search_text = sender.text().lower()  # Get the search text and convert to lower case

        if sender == self.covered_by_search_bar:
            # Filter covered by tree
            root = self.covered_by_tree.invisibleRootItem()
            for i in range(root.childCount()):
                item = root.child(i)
                self.filter_item(item, search_text)  # Use existing filter_item method

        elif sender == self.joining_search_bar:
            # Filter joining tree
            for i in range(self.joining_tree.topLevelItemCount()):
                item = self.joining_tree.topLevelItem(i)
                self.filter_item(item, search_text)  # Use existing filter_item method
        
        elif sender == self.classification_search_bar:
            # Filter classification tree
            for i in range(self.classification_tree.topLevelItemCount()):
                item = self.classification_tree.topLevelItem(i)
                self.filter_item(item, search_text)
        
        elif sender == self.connected_with_search_bar:
            # Filter connected with tree
            for i in range(self.connected_with_tree.topLevelItemCount()):
                item = self.connected_with_tree.topLevelItem(i)
                self.filter_item(item, search_text)
        
        elif sender == self.joining_component_search_bar:
            # Filter connected with tree
            for i in range(self.joining_component_tree.topLevelItemCount()):
                item = self.joining_component_tree.topLevelItem(i)
                self.filter_item(item, search_text)


    def filter_item(self, item, search_text):
        """
        Recursively filter items in the tree.

        This method checks if the item's text or any of its children's text matches the search text.
        If a match is found, the item is shown; otherwise, it is hidden.

        Args:
            item (QTreeWidgetItem): The tree item to check.
            search_text (str): The search text to match against the item's text.

        Returns:
            bool: `True` if the item or any of its children matches the search text, `False` otherwise.

        Notes:
            - If the item's text matches the search text, all its children are shown.
            - The method recursively checks all child items.
        """
        item_text = item.text(0).lower()
        match = search_text in item_text

        # Check children
        child_match = False
        for i in range(item.childCount()):
            child = item.child(i)
            child_match = self.filter_item(child, search_text) or child_match

        # If parent matches, show all children and their children
        if match:
            self.show_all_children(item)

        # Show or hide item based on match or child match
        item.setHidden(not (match or child_match))
        return match or child_match

    def show_all_children(self, item):
        """
        Recursively show all children of the given item.

        This method ensures that all child items of the given tree item are made visible.

        Args:
            item (QTreeWidgetItem): The tree item whose children should be shown.

        Returns:
            None

        Notes:
            - This method is used in conjunction with filtering to ensure that matching
            child items are visible even if their parent is hidden.
        """
        for i in range(item.childCount()):
            child = item.child(i)
            child.setHidden(False)
            self.show_all_children(child)

    def save_data(self):
        """
        Saves the entered data back to the JSON file after validating all parameters.

        This method collects data from the user interface for each category, validates the parameters,
        and updates the JSON data. If validation fails, the save operation is aborted.

        Args:
            None

        Returns:
            None

        Notes:
            - The method loads category data from the JSON file to determine the parameters for each category.
            - Data for each category is collected using the `collect_category_data` method.
            - Validation is performed using the `validate_all_parameters` method.
            - If validation passes, the JSON data is updated using the `update_json_data` method.
            - The dialog is closed after successfully saving the data.
        """
        # Load the existing category data
        category_data = load_json_file(self.category_path)
        categories = category_data.get('categories', {})
        self.param_mapping = category_data.get('param_mapping', {})

        # Collect data for each category
        for category in categories.keys():
            self.collect_category_data(category)

        # Validate all parameters
        if not self.validate_all_parameters(categories):
            return  # Stop saving if validation fails


        self.update_json_data()  # Update the JSON data
        self.accept()  # Close the dialog after saving

    
    def validate_all_parameters(self, categories):
        """
        Validates all parameters and ensures they are assessed.

        This method checks the completeness and validity of parameters for each category
        (e.g., Material, Development, Basic Parameters, Disassembly). If any parameter is
        missing or invalid, an error message is displayed, and the validation fails.

        Args:
            categories (dict): A dictionary of categories and their parameters.

        Returns:
            bool: `True` if all parameters are valid and assessed, `False` otherwise.

        Notes:
            - For the "Material" category, it validates material parameters, fractions, and database entries.
            - For the "Development" category, it ensures all parameters are selected.
            - For the "Basic Parameters" category, it ensures all fields are filled, "classification" exists and the value of "lifespan [a]" is numeric.
            - For the "Disassembly" category, it validates joining components and connections.
            - If validation fails, a critical error message is displayed with details.
        """
        has_error = False
        error_messages = []

        for category in categories.keys():
            if category == "Material":
                if "material_parameters" not in self.item_data or not self.item_data["material_parameters"]:
                    has_error = True
                    error_messages.append("Material parameters are not fully assessed.")
                else:
                    # Validate number_of_materials
                    if "number_of_materials" not in self.item_data["material_parameters"] or \
                        self.item_data["material_parameters"]["number_of_materials"] <= 0:
                        has_error = True
                        error_messages.append("Number of materials is not valid.")

                    # Validate material_fractions
                    if "material_fractions" not in self.item_data["material_parameters"] or \
                        not self.item_data["material_parameters"]["material_fractions"]:
                        has_error = True
                        error_messages.append("Material fractions are missing.")
                    
                    else:
                        # Validate each material in material_fractions
                        for material in self.item_data["material_parameters"]["material_fractions"]:
                            material_name = material.get("type")
                            if not material_name:
                                has_error = True
                                error_messages.append("A material type is missing in material fractions.")
                                continue

                            # Check if the material is in the database or has set parameters
                            if material_name not in self.materials_db.get("materials", {}) and \
                                material_name not in self.material_parameters:
                                has_error = True
                                error_messages.append(f"Material '{material_name}' is not found in the database.")

            elif category == "Development":
                if "development_parameters" not in self.item_data or not self.item_data["development_parameters"]:
                    has_error = True
                    error_messages.append("Development parameters are not fully assessed.")
                else:
                    for param, value in self.item_data["development_parameters"].items():
                        if value == "" or value is None:
                            has_error = True
                            error_messages.append(f"Development parameter '{param}' is not selected.")

            elif category == "Basic Parameters":
                if "basic_parameters" not in self.item_data or not self.item_data["basic_parameters"]:
                    has_error = True
                    error_messages.append("Basic parameters are not fully assessed.")
                else:
                    for param, value in self.item_data["basic_parameters"].items():
                        if value == "" or value is None or value == "None":
                            has_error = True
                            error_messages.append(f"Basic parameter '{param}' is empty.")
                    if "classification" not in self.item_data["basic_parameters"] or not self.item_data["basic_parameters"]["classification"]:
                        has_error = True
                        error_messages.append("Basic parameter classification is empty.")
                    try:
                        if int(self.item_data["basic_parameters"]["lifespan [a]"]) <= 0:
                            has_error = True
                            error_messages.append("Basic parameter lifespan is zero or negative.")
                    except ValueError:
                        has_error = True
                        error_messages.append("Please fill in the basic parameter lifespan as a number.")

            elif category == "Disassembly":
                if "disassembly_parameters" not in self.item_data or not self.item_data["disassembly_parameters"]:
                    has_error = True
                    error_messages.append("Disassembly parameters are not fully assessed.")
                else:
                    disassembly_params = self.item_data["disassembly_parameters"]
                    # Check if "is_joining_component" is present
                    if "is_joining_component" not in disassembly_params:
                        has_error = True
                        error_messages.append("Disassembly parameter 'is_joining_component' is not assessed.")
                    else:
                        # If "is_joining_component" is False, validate "assigned_connections"
                        if not disassembly_params["is_joining_component"]:
                            if "assigned_connections" not in disassembly_params or not disassembly_params["assigned_connections"]:
                                has_error = True
                                error_messages.append("Disassembly parameter 'assigned_connections' is not assessed.")

                    # Validate "joining_components" if "is_joining_component" is True
                    if disassembly_params.get("is_joining_component", False):
                        if "joining_components" not in disassembly_params or not disassembly_params["joining_components"]:
                            has_error = True
                            error_messages.append("Disassembly parameter 'joining_components' is not assessed.")
            
            else: 
                if "assessed" not in self.item_data or not self.item_data["assessed"]:
                    has_error = True
                    error_messages.append("Please select 'Assessed' to confirm that all parameters are assessed.")

        # Show error messages if validation fails
        if has_error:
            QMessageBox.critical(self, "Validation Error", "\n".join(error_messages))
        return not has_error
    
    def collect_category_data(self, category):
        """
        Collects data for a specific category and updates item_data.

        This method collects user-entered data for a given category (e.g., Material, Development, 
        Basic Parameters, Disassembly) and updates the `item_data` dictionary with the collected data.

        Args:
            category (str): The name of the category to collect data for.

        Returns:
            None

        Notes:
            - For predefined categories (Material, Development, Basic Parameters, Disassembly), 
            specific methods are called to collect data.
            - For other categories, the method dynamically handles data collection by locating 
            the corresponding group box in the UI.
            - The 'assessed' state is also collected and stored in `item_data`.
        """
        if category == "Material":
            self.collect_material_parameters()
            if hasattr(self, "material_parameters"):
                self.item_data["material_parameters"] = self.material_parameters

        elif category == "Development":
            self.collect_development_parameters()
            if hasattr(self, "development_parameters"):
                self.item_data["development_parameters"] = self.development_parameters

        elif category == "Basic Parameters":
            self.collect_basic_parameters()
            if hasattr(self, "basic_parameters"):
                self.item_data["basic_parameters"] = self.basic_parameters

        elif category == "Disassembly":
            self.collect_disassembly_parameters()
            if hasattr(self, "disassembly_parameters"):
                self.item_data["disassembly_parameters"] = self.disassembly_parameters

        else:
            # Handle other categories dynamically
            group_box = self.findChild(QGroupBox, category)
            if not group_box:
                print(f"Group box for category '{category}' not found. Skipping.")
                return
        
        # Collect the 'assessed' state
        self.item_data['assessed'] = self.assessed_checkbox.isChecked()
    
    def collect_basic_parameters(self):
        """
        Collect data for the 'Basic Parameters' category.

        This method collects user-entered data from the widgets associated with the 'Basic Parameters'
        category and updates the `basic_parameters` dictionary.

        Args:
            None

        Returns:
            None

        Notes:
            - The method handles various widget types, including QComboBox, QLineEdit, QSpinBox, and QCheckBox.
            - The classification data is collected from the `classification_tree` and stored in the `classification` key.
            - If no classification was selected the exception raised prevents the app from crashing.
        """
        for param in self.basic_param_widgets:
            widget = self.basic_param_widgets[param]
            if isinstance(widget, QComboBox):
                self.basic_parameters[param] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                self.basic_parameters[param] = widget.text()
            elif isinstance(widget, QSpinBox):
                self.basic_parameters[param] = widget.value()
            elif isinstance(widget, QCheckBox):
                self.basic_parameters[param] = widget.isChecked()

        try:
            self.classfication_item = self.get_checked_items_from_tree(self.classification_tree)
            self.basic_parameters["classification"] = self.classfication_item[0]
        
        except:
            QMessageBox.critical(self, "Validation Error", "Please select a classification.")
            return

    def collect_disassembly_parameters(self):
        """
        Collect all disassembly-related parameters and store them in disassembly_parameters.

        This method collects user-entered data related to disassembly, including assigned connections,
        joining component status, covered-by components, and joining components, and updates the
        `disassembly_parameters` dictionary.

        Args:
            None

        Returns:
            None

        Notes:
            - The `assigned_connections` are collected if available.
            - The state of the `is_joining_component_checkbox` is stored in `is_joining_component`.
            - Components covered by the current component are collected from the `covered_by_tree`.
            - Checked items from the `joining_component_tree` are stored as joining components.
        """

        # Collect assigned connections
        if hasattr(self, "assigned_connections"):
            self.disassembly_parameters["assigned_connections"] = self.assigned_connections

        # Collect is_joining_component checkbox state
        if hasattr(self, "is_joining_component_checkbox"):
            self.disassembly_parameters["is_joining_component"] = self.is_joining_component_checkbox.isChecked()
        
        # Collect covered by components
        self.covered_by = self.get_checked_items_from_tree(self.covered_by_tree)
        if hasattr(self, "covered_by"):
            self.disassembly_parameters["covered_by"] = self.covered_by
        
        # Collect checked items from the joining_component_tree
        self.is_joining = self.get_checked_items_from_tree(self.joining_component_tree)
        if hasattr(self, "is_joining"):
            self.disassembly_parameters["joining_components"] = self.is_joining
    
    def collect_material_parameters(self):
        """
        Collects all material-related parameters.
        - number_of_materials = 1
        - material_fractions = list of {type, percentage}
        - material_parameters = dict with current material (even if empty)

        """
        # Ensure self.material_parameters exists
        if not hasattr(self, "material_parameters"):
            self.material_parameters = {}

        # Number of materials = 1
        self.material_parameters["number_of_materials"] = 1


        # Material fractions (unchanged)
        material_fractions, _, _ = self.calculate_material_fractions_components(self.item_data)
        self.material_parameters["material_fractions"] = material_fractions

        # Always store current material (even if empty)
        if hasattr(self, "material_combo"):
            mat_key = self.material_combo.currentData()
            if mat_key:
                # Ensure entry exists (even if empty)
                if mat_key not in self.material_parameters:
                    self.material_parameters[mat_key] = {}
                self.material_parameters["assessed_material"] = mat_key

    def collect_development_parameters(self):
        """
        Collect data for the 'Development' category.

        This method collects user-entered data from the widgets associated with the 'Development'
        category and updates the `development_parameters` dictionary.

        Args:
            None

        Returns:
            None

        Notes:
            - The method handles various widget types, including QComboBox, QLineEdit, QSpinBox, and QCheckBox.
            - The collected data is stored in the `development_parameters` dictionary.
        """
        for param in self.development_param_widgets:
            widget = self.development_param_widgets[param]
            if isinstance(widget, QComboBox):
                self.development_parameters[param] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                self.development_parameters[param] = widget.text()
            elif isinstance(widget, QSpinBox):
                self.development_parameters[param] = widget.value()
            elif isinstance(widget, QCheckBox):
                self.development_parameters[param] = widget.isChecked()

    def update_json_data(self):
        """
        Updates the JSON data with the item data and saves it to the file.

        This method loads the current structure from the JSON file, updates the relevant
        item data in the structure, and saves the updated structure back to the file.

        Args:
            None

        Returns:
            None

        Notes:
            - The current structure is loaded from the file specified by `self.structure_file_path`.
            - The `update_json_structure` method is used to update the item data in the structure.
            - If a connection was deleted the `mirror_empty_assigned_connections_and_delete` method is used to mirror the empty assigned connections.
            - The `synchronize_assigned_connections_by_richer_side` method is used to copy the payload with more connection types to both sides for each component pair. 
            - The updated structure is saved back to the file using the `dump_file` function.
        """

        # Load the current structure file
        data = load_json_file(self.structure_file_path)

        # Update the item data in the structure
        self.update_json_structure(data, self.item_data)
        
        if self.deleted_connection == True:
            mirror_empty_assigned_connections_and_delete(data)
            self.deleted_connection = False

        synchronize_assigned_connections_by_richer_side(data)

        # Save the updated structure back to the file
        dump_file(data, self.structure_file_path)

    def update_json_structure(self, data, updated_item):
        """
        Recursively updates the JSON data with the updated_item based on matching 'id'.

        This method searches through the JSON structure to find an item with a matching 'id'
        and updates its keys with the values from the `updated_item`.

        Args:
            data (dict or list): The JSON data structure to update.
            updated_item (dict): The item containing updated data, including its 'id'.

        Returns:
            None

        Notes:
            - If no 'id' is found in the `updated_item`, the method prints a warning and exits.
            - The method updates only the keys present in the `updated_item`.
            - The search is performed recursively to handle nested structures.
            - Once the matching item is updated, the recursion stops.
        """
        target_id = updated_item.get('id')
        if not target_id:
            print("No 'id' in updated_item. Cannot update.")
            return

        # Recursively search for the item with the matching 'id'
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get('id') == target_id:
                    # Update only the relevant keys in the matching item
                    for key, value in updated_item.items():
                        item[key] = value
                    return  # Stop further recursion once the item is updated
                else:
                    # Recursively search in nested structures
                    self.update_json_structure(item, updated_item)

        elif isinstance(data, dict):
            if data.get('id') == target_id:
                # Update only the relevant keys in the matching item
                for key, value in updated_item.items():
                    data[key] = value
                return  # Stop further recursion once the item is updated
            else:
                # Recursively check nested dictionaries or lists
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        self.update_json_structure(value, updated_item)

class JoiningConnectionDialog(QDialog):
    """
    A dialog for assigning details to a joining connection.

    This class provides a user interface for entering details about a joining connection,
    such as the amount, whether it is non-destructive, ageing status, tool used, and detachment time.

    Attributes:
        amount_spinbox (QSpinBox): Input field for specifying the amount of the connection.
        non_destructive_combo (QComboBox): Dropdown for selecting if the connection is non-destructive.
        ageing_combo (QComboBox): Dropdown for selecting if the connection is affected by ageing.
        tool_combo (QComboBox): Dropdown for selecting the tool used for the connection.
        detachment_time_spinbox (QSpinBox): Input field for specifying the detachment time in seconds.

    Methods:
        __init__(connection_name, parent=None): Initializes the dialog with the given connection name.
    """
    def __init__(self, connection_name, parent=None):
        """
        Initialize the JoiningConnectionDialog.

        This constructor sets up the dialog's user interface, including input fields for
        connection details and OK/Cancel buttons.

        Args:
            connection_name (str): The name of the joining connection being assigned.
            parent (QWidget, optional): The parent widget of the dialog. Defaults to None.

        Returns:
            None
        """
        super().__init__(parent)

        self.tools_json =  BASE_DIR / "disassembly_tools.json"
        self.setWindowTitle(f"Assign Joining Connection: {connection_name}")

        # Layout
        layout = QVBoxLayout(self)

        # Form layout for input fields
        form_layout = QFormLayout()

        # Amount input
        self.amount_spinbox = QSpinBox()
        self.amount_spinbox.setRange(1, 1000)
        self.amount_spinbox.setValue(1)
        form_layout.addRow("Amount:", self.amount_spinbox)

        # Non-destructive detachable input
        self.non_destructive_combo = QComboBox()
        self.non_destructive_combo.addItems(["Yes", "No"])
        form_layout.addRow("Non-Destructive Detachable:", self.non_destructive_combo)

        # Ageing input
        self.ageing_combo = QComboBox()
        self.ageing_combo.addItems(["Yes", "No"])
        form_layout.addRow("Ageing:", self.ageing_combo)

        # Tool used input
        self.tool_combo = QComboBox()
        self.tool_combo.setPlaceholderText("Enter tool used (if any)")
        tool_data = load_json_file(self.tools_json)
        tools = tool_data.get("tools", [])
        self.tool_combo.addItems(tools)
        form_layout.addRow("Tool Used:", self.tool_combo)

        # Detachment time input
        self.detachment_time_spinbox = QSpinBox()
        self.detachment_time_spinbox.setRange(1, 1000)  # Time in seconds
        self.detachment_time_spinbox.setValue(10)
        form_layout.addRow("Detachment Time (seconds):", self.detachment_time_spinbox)

        layout.addLayout(form_layout)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_data(self):
        """
        Return the data entered by the user.

        This method collects the details entered by the user in the `JoiningConnectionDialog`
        and returns them as a dictionary.

        Args:
            None

        Returns:
            dict: A dictionary containing the following keys:
                - "amount" (int): The specified amount of the connection.
                - "non_destructive" (bool): Whether the connection is non-destructive.
                - "ageing" (bool): Whether the connection is affected by ageing.
                - "tool" (str): The tool used for the connection.
                - "detachment_time" (int): The detachment time in seconds.
        """
        return {
            "amount": self.amount_spinbox.value(),
            "non_destructive": self.non_destructive_combo.currentText() == "Yes",
            "ageing": self.ageing_combo.currentText() == "Yes",
            "tool": self.tool_combo.currentText(),
            "detachment_time": self.detachment_time_spinbox.value(),
        }

class MaterialParameterDialog(QDialog):
    """
    A dialog for entering material parameters.

    This class provides a user interface for entering detailed parameters about a material,
    such as density, recycling percentage, recyclate content, and other material-specific attributes.

    Attributes:
        density_input (QLineEdit): Input field for specifying the material's density.
        recycling_spinbox (QSpinBox): Input field for specifying the recycling percentage.
        recyclate_spinbox (QSpinBox): Input field for specifying the recyclate content percentage.
        recycling_criticality_combo (QComboBox): Dropdown for selecting if recycling is critical.
        incineration_for_energy_recovery_combo (QComboBox): Dropdown for selecting if incineration can recover energy.
        environmental_harmfulness_combo (QComboBox): Dropdown for selecting if the material is environmentally harmful.
        health_harmfulness_combo (QComboBox): Dropdown for selecting if the material is harmful to health.
        monomaterial_combo (QComboBox): Dropdown for selecting if the material is monomaterial.
        additives_combo (QComboBox): Dropdown for selecting if the material contains additives or fillers.
        surface_coatings_combo (QComboBox): Dropdown for selecting if the material has surface coatings.
        availability_combo (QComboBox): Dropdown for specifying the material's availability.

    Methods:
        __init__(material_name, parent=None): Initializes the dialog with the given material name.
    """
    def __init__(self, material_name, parent=None):
        """
        Initialize the MaterialParameterDialog.

        This constructor sets up the dialog's user interface, including input fields for
        material parameters and OK/Cancel buttons.

        Args:
            material_name (str): The name of the material being assessed.
            parent (QWidget, optional): The parent widget of the dialog. Defaults to None.

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle(f"Material Parameters: {material_name}")

        # Layout
        layout = QVBoxLayout(self)

        # Form layout for input fields
        form_layout = QFormLayout()

        # Density input (using QLineEdit with QDoubleValidator)
        self.density_input = QLineEdit()
        self.density_input.setPlaceholderText("Enter density")
        form_layout.addRow("Density (kg/mm^3)", self.density_input)
            
        # Recycling percentage
        self.recycling_spinbox = QSpinBox()
        self.recycling_spinbox.setRange(0, 100)
        self.recycling_spinbox.setValue(0)
        form_layout.addRow("Recycling (%)", self.recycling_spinbox)

        # Recyclate content percentage
        self.recyclate_spinbox = QSpinBox()
        self.recyclate_spinbox.setRange(0, 100)
        self.recyclate_spinbox.setValue(0)
        form_layout.addRow("Recyclate Content (%)", self.recyclate_spinbox)

        # Yes/No questions
        self.recycling_criticality_combo = QComboBox()
        self.recycling_criticality_combo.addItems(["Yes", "No"])
        form_layout.addRow("Recycling Criticality", self.recycling_criticality_combo)

        self.incineration_for_energy_recovery_combo = QComboBox()
        self.incineration_for_energy_recovery_combo.addItems(["Yes", "No"])
        self.incineration_for_energy_recovery_combo.setCurrentText("No")
        form_layout.addRow("Incineration for energy recovery", self.incineration_for_energy_recovery_combo)

        

        self.environmental_harmfulness_combo = QComboBox()
        self.environmental_harmfulness_combo.addItems(["Yes", "No"])
        form_layout.addRow("Environmental Harmfulness", self.environmental_harmfulness_combo)

        self.health_harmfulness_combo = QComboBox()
        self.health_harmfulness_combo.addItems(["Yes", "No"])
        form_layout.addRow("Harmfulness for Health", self.health_harmfulness_combo)

        self.monomaterial_combo = QComboBox()
        self.monomaterial_combo.addItems(["Yes", "No"])
        form_layout.addRow("Monomaterial", self.monomaterial_combo)

        self.additives_combo = QComboBox()
        self.additives_combo.addItems(["Yes", "No"])
        form_layout.addRow("Contains Additives or Fillers", self.additives_combo)

        self.surface_coatings_combo = QComboBox()
        self.surface_coatings_combo.addItems(["Yes", "No"])
        form_layout.addRow("Surface Coatings", self.surface_coatings_combo)

        self.availability_combo = QComboBox()
        self.availability_combo.addItems(["Regionally", "Nationally", "Same continent", "Other continent"])
        form_layout.addRow("Availability", self.availability_combo)

        layout.addLayout(form_layout)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_data(self):
        """
        Return the data entered by the user.

        This method collects the details entered by the user in the `MaterialParameterDialog`
        and returns them as a dictionary.

        Args:
            None

        Returns:
            dict: A dictionary containing the following keys:
                - "recycling_percentage" (int): The recycling percentage.
                - "recyclate_content" (int): The recyclate content percentage.
                - "density [kg/mm^3]" (float): The density in scientific notation.
                - "recycling_criticality" (bool): Whether recycling is critical.
                - "incineration_for_energy_recovery" (bool): Whether the material can be incinerated for energy recovery.
                - "environmental_harmfulness" (bool): Whether the material is environmentally harmful.
                - "health_harmfulness" (bool): Whether the material is harmful to health.
                - "monomaterial" (bool): Whether the material is monomaterial.
                - "additives_or_fillers" (bool): Whether the material contains additives or fillers.
                - "surface_coatings" (bool): Whether the material has surface coatings.
                - "availability" (str): The availability of the material (e.g., "Regionally", "Nationally", "Same continent", "Other continent").
        """
        # Convert density to scientific notation
        density_text = self.density_input.text()
        density_value = float(density_text) if density_text else 0.0
        density_scientific = float(f"{density_value:.2e}")  # Format as scientific notation and convert to float

        return {
            "recycling_percentage": self.recycling_spinbox.value(),
            "recyclate_content": self.recyclate_spinbox.value(),
            "density [kg/mm^3]": density_scientific,  # Store density in scientific notation
            "recycling_criticality": self.recycling_criticality_combo.currentText() == "Yes",
            "incineration_for_energy_recovery": self.incineration_for_energy_recovery_combo.currentText() == "Yes",
            "environmental_harmfulness": self.environmental_harmfulness_combo.currentText() == "Yes",
            "health_harmfulness": self.health_harmfulness_combo.currentText() == "Yes",
            "monomaterial": self.monomaterial_combo.currentText() == "Yes",
            "additives_or_fillers": self.additives_combo.currentText() == "Yes",
            "surface_coatings": self.surface_coatings_combo.currentText() == "Yes",
            "availability": self.availability_combo.currentText(),
        }
    
    def accept(self):
        """
        Validate the input and close the dialog if valid.

        This method validates the user-entered density value in the `MaterialParameterDialog`.
        If the input is invalid (e.g., contains a comma or is not a positive number), an error
        message is displayed, and the dialog remains open. If the input is valid, the dialog is closed.

        Args:
            None

        Returns:
            None

        Notes:
            - The density input must use a point (.) as the decimal separator.
            - The density value must be a positive number.
            - If validation fails, a critical error message is displayed.
        """
        density_text = self.density_input.text()
        if ',' in density_text:
            QMessageBox.critical(self, "Invalid Input", "Please use a point (.) instead of a comma (,) for the density.")
            return  # Prevent the dialog from closing

        # Proceed with other validations
        try:
            density_value = float(density_text)
            if density_value <= 0:
                raise ValueError("Density must be greater than 0.")
        except ValueError:
            QMessageBox.critical(self, "Invalid Input", "Please enter a valid numeric density.")
            return  # Prevent the dialog from closing

        super().accept()  # Close the dialog if input is valid

class MaterialParameterAssessmentDialog(QDialog):
    """
    A dialog for assessing material parameters.

    This class provides a user interface for changing detailed parameters about a material,
    such as recycling percentage, recyclate content, and other material-specific attributes.

    Attributes:
        density_label (QLabel): Field for showing the material's density.
        recycling_spinbox (QSpinBox): Input field for specifying the recycling percentage.
        recyclate_spinbox (QSpinBox): Input field for specifying the recyclate content percentage.
        recycling_criticality_combo (QComboBox): Dropdown for selecting if recycling is critical.
        incineration_for_energy_recovery_combo (QComboBox): Dropdown for selecting if incineration can recover energy.
        environmental_harmfulness_combo (QComboBox): Dropdown for selecting if the material is environmentally harmful.
        health_harmfulness_combo (QComboBox): Dropdown for selecting if the material is harmful to health.
        monomaterial_combo (QComboBox): Dropdown for selecting if the material is monomaterial.
        additives_combo (QComboBox): Dropdown for selecting if the material contains additives or fillers.
        surface_coatings_combo (QComboBox): Dropdown for selecting if the material has surface coatings.
        availability_combo (QComboBox): Dropdown for specifying the material's availability.

    Methods:
        __init__(material_name, parent=None): Initializes the dialog with the given material name.
    """
    def __init__(self, material_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Material Parameters: {material_name}")
        self.material_name = material_name

        # Load materials database
        materials_db = {}
        try:
            parent_config = parent.config if hasattr(parent, 'config') else None
            if parent_config:
                materials_path = parent_config.materials_json
            else:
                materials_path = "app_data/materials_validation.json"
            with open(materials_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                materials_db = data.get("materials", {})
        except Exception as e:
            print(f"⚠️ Materialien konnten nicht geladen werden: {e}")

        # Get material data (existing from DB or default)
        self.material_data = materials_db.get(material_name, {})
        self.density = self.material_data.get("density [kg/mm^3]")

        # Layout
        layout = QVBoxLayout(self)

        # Form layout for input fields
        form_layout = QFormLayout()

                # Density — READ-ONLY, auto-filled
        self.density_label = QLabel()
        if self.density is not None:
            self.density_label.setText(f"{self.density:.2e} kg/mm³")
        else:
            self.density_label.setText("—")
        form_layout.addRow("Density (kg/mm³):", self.density_label)

        # Recycling percentage (only if not None in DB)
        self.recycling_spinbox = QSpinBox()
        self.recycling_spinbox.setRange(0, 100)
        val = self.material_data.get("recycling_percentage")
        if val is not None:
            self.recycling_spinbox.setValue(val)
        form_layout.addRow("Recycling (%):", self.recycling_spinbox)


        # Recyclate content percentage
        self.recyclate_spinbox = QSpinBox()
        self.recyclate_spinbox.setRange(0, 100)
        val = self.material_data.get("recyclate_content")
        if val is not None:
            self.recyclate_spinbox.setValue(val)
        form_layout.addRow("Recyclate Content (%):", self.recyclate_spinbox)

        # Yes/No questions (only show if present in DB)
        self.recycling_criticality_combo = QComboBox()
        self.recycling_criticality_combo.addItems(["Yes", "No"])
        val = self.material_data.get("recycling_criticality")
        if val is not None:
            self.recycling_criticality_combo.setCurrentText("Yes" if val else "No")
        form_layout.addRow("Recycling Criticality:", self.recycling_criticality_combo)

        self.incineration_for_energy_recovery_combo = QComboBox()
        self.incineration_for_energy_recovery_combo.addItems(["Yes", "No"])
        val = self.material_data.get("incineration_for_energy_recovery", False)
        self.incineration_for_energy_recovery_combo.setCurrentText("Yes" if val else "No")
        form_layout.addRow("Incineration for energy recovery:", self.incineration_for_energy_recovery_combo)


        self.environmental_harmfulness_combo = QComboBox()
        self.environmental_harmfulness_combo.addItems(["Yes", "No"])
        val = self.material_data.get("environmental_harmfulness")
        if val is not None:
            self.environmental_harmfulness_combo.setCurrentText("Yes" if val else "No")
        form_layout.addRow("Environmental Harmfulness:", self.environmental_harmfulness_combo)

        self.health_harmfulness_combo = QComboBox()
        self.health_harmfulness_combo.addItems(["Yes", "No"])
        val = self.material_data.get("health_harmfulness")
        if val is not None:
            self.health_harmfulness_combo.setCurrentText("Yes" if val else "No")
        form_layout.addRow("Harmfulness for Health:", self.health_harmfulness_combo)

        self.monomaterial_combo = QComboBox()
        self.monomaterial_combo.addItems(["Yes", "No"])
        val = self.material_data.get("monomaterial")
        if val is not None:
            self.monomaterial_combo.setCurrentText("Yes" if val else "No")
        form_layout.addRow("Monomaterial:", self.monomaterial_combo)

        self.additives_combo = QComboBox()
        self.additives_combo.addItems(["Yes", "No"])
        val = self.material_data.get("additives_or_fillers")
        if val is not None:
            self.additives_combo.setCurrentText("Yes" if val else "No")
        form_layout.addRow("Contains Additives or Fillers:", self.additives_combo)

        self.surface_coatings_combo = QComboBox()
        self.surface_coatings_combo.addItems(["Yes", "No"])
        val = self.material_data.get("surface_coatings")
        if val is not None:
            self.surface_coatings_combo.setCurrentText("Yes" if val else "No")
        form_layout.addRow("Surface Coatings:", self.surface_coatings_combo)

        self.availability_combo = QComboBox()
        self.availability_combo.addItems(["Regionally", "Nationally", "Same continent", "Other continent"])
        val = self.material_data.get("availability")
        if val:
            self.availability_combo.setCurrentText(val)
        form_layout.addRow("Availability:", self.availability_combo)

        layout.addLayout(form_layout)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_assessment_data(self):
        """
        Return only the editable (assessment) parameters — density is excluded.

        Returns:
            dict: Keys — recycling_percentage, recyclate_content,
                recycling_criticality, incineration_for_energy_recovery, environmental_harmfulness,
                health_harmfulness, monomaterial, additives_or_fillers,
                surface_coatings, availability.

        """
        return {
            "recycling_percentage": self.recycling_spinbox.value(),
            "recyclate_content": self.recyclate_spinbox.value(),
            "recycling_criticality": self.recycling_criticality_combo.currentText() == "Yes",
            "incineration_for_energy_recovery": self.incineration_for_energy_recovery_combo.currentText() == "Yes",
            "environmental_harmfulness": self.environmental_harmfulness_combo.currentText() == "Yes",
            "health_harmfulness": self.health_harmfulness_combo.currentText() == "Yes",
            "monomaterial": self.monomaterial_combo.currentText() == "Yes",
            "additives_or_fillers": self.additives_combo.currentText() == "Yes",
            "surface_coatings": self.surface_coatings_combo.currentText() == "Yes",
            "availability": self.availability_combo.currentText(),
        }

class CalculationDialog(QDialog):
    """
    A dialog for initiating and managing calculations.

    This class provides a user interface for starting calculations based on the provided
    data, configuration, and file paths. It loads necessary data files and initializes
    the UI for user interaction.

    Attributes:
        data (dict): The hierarchical structure data for the calculation.
        structure_file_path (str): The file path to the structure JSON file.
        material_file_path (str): The file path to the materials JSON file.
        material_data (dict): The loaded material data from the materials JSON file.
        parameter_to_goal_file_path (str): The file path to the parameter-to-goal mapping JSON file.
        parameter_to_goal (dict): The loaded parameter-to-goal mapping data.
        goal_to_r_strategy_file_path (str): The file path to the goal-to-R-strategy mapping JSON file.
        goal_to_r_strategy (dict): The loaded goal-to-R-strategy mapping data.
        joining_json_path (str): The file path to the joining connections JSON file.
        classification_json (str): The file path to the classification JSON file.
        classification_data (dict): The loaded classification data.

    Methods:
        __init__(data, file, config: Config, parent=None): Initializes the dialog with the given data and configuration.
        init_ui(): Sets up the user interface for the dialog.
    """

    def __init__(self, data, file, config: Config, parent=None):
        """
        Initialize the CalculationDialog.

        This constructor sets up the dialog with the provided data, file paths, and configuration.
        It also loads necessary JSON files and initializes the user interface.

        Args:
            data (dict): The hierarchical structure data for the calculation.
            file (str): The file path to the structure JSON file.
            config (Config): An instance of the Config class containing configuration settings.
            parent (QWidget, optional): The parent widget of the dialog. Defaults to None.

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle("Start Calculation")
        self.data = data
        self.structure_file_path = file
        self.material_file_path = config.materials_json
        self.material_data = load_json_file(self.material_file_path)["materials"]
        self.parameter_to_goal_file_path = config.parameter_to_goal_json
        self.parameter_to_goal = load_json_file(self.parameter_to_goal_file_path)
        self.goal_to_r_strategy_file_path = config.goal_to_r_strategy_json
        self.goal_to_r_strategy = load_json_file(self.goal_to_r_strategy_file_path)
        self.joining_json_path = config.joining_connections_json
        self.classification_json = config.classification_json
        self.classification_data = load_json_file(self.classification_json)
        self.init_ui()
    
    def init_ui(self):
        """
        Initializes the UI with a Start Calculation button, a status bar, and a Show Results button.

        This method sets up the user interface for the `CalculationDialog`, including buttons
        for starting calculations and showing results, as well as a status bar to display the
        current status.

        Args:
            None

        Returns:
            None

        Notes:
            - The Start Calculation button is connected to the `start_calculation` method.
            - The Show Results button is connected to the `show_results` method.
            - The status bar displays the current status of the dialog.
        """
        layout = QVBoxLayout()

        # Add the Start Calculation button
        self.start_button = QPushButton("Start Calculation")
        self.start_button.clicked.connect(self.start_calculation)  # Connect to the calculation method
        layout.addWidget(self.start_button)

        # Add the status bar
        self.status_bar = QLabel("Status: Ready")
        self.status_bar.setStyleSheet("padding: 5px; border: 1px solid #ccc; background-color: #f9f9f9;")
        layout.addWidget(self.status_bar)

        # Add the Show Results button
        self.show_results_button = QPushButton("Show Results")
        self.show_results_button.clicked.connect(self.show_results)  # Connect to the method to show results
        layout.addWidget(self.show_results_button)

    
        # Set the layout for the dialog
        self.setLayout(layout)

    def start_calculation(self):
        """
        Handles all the calculation methods and updates the status bar.

        This method orchestrates the execution of various calculation methods, updating the status bar
        at each step to reflect the progress. It also handles exceptions and displays error messages
        if any issues occur during the calculations.

        Args:
            None

        Returns:
            None

        Notes:
            - The method calculates various parameters, including classification averages, number of components,
            material types, joining connections, and other metrics.
            - It collects unique materials, calculates binary material parameters, and computes goal achievements
            and R-strategy scores.
            - The status bar is updated after each calculation step to provide feedback to the user.
            - If an error occurs, the status bar is updated with the error message, and a critical error dialog is displayed.
        """
        self.status_bar.setText("Status: Starting calculations...\n")
        self.all_parameters = {}
        self.goal_scores = {}
        self.r_strategy_scores = {}

        try:
            validate_goal_weights(self.goal_to_r_strategy)
        except ValueError as exc:
            self.status_bar.setText("Status: Calculation stopped. Correct the goal-to-R-strategy weights.")
            QMessageBox.warning(self, "Invalid goal weights", str(exc))
            return

        try:
            # Derive the assembly classification and validate its reference examples first.
            if not self.calculate_classification_averages():
                self.status_bar.setText(
                    "Status: Calculation stopped. Check part classifications and assembly reference examples."
                )
                return
            current_status = self.status_bar.text()
            self.status_bar.setText(current_status + "Classification averages calculated.\n")

            # Calculate number of components
            self.calculate_number_of_components()
            current_status = self.status_bar.text()
            self.status_bar.setText(current_status + f"Number of components calculated: {self.number_of_components}\n")

            # Calculate number of components
            self.calculate_number_of_material_types()
            current_status = self.status_bar.text()
            self.status_bar.setText(current_status + f"Number of material types calculated: {len(self.material_list)}\n")

            #Calculate the number of joining connections
            self.calculate_joining_connections()
            current_status = self.status_bar.text()
            self.status_bar.setText(current_status + f"Number of joining connections calculated: {self.number_of_joining_connections}\n")
            current_status = self.status_bar.text()
            self.status_bar.setText(current_status + f"Total disassembly time calculated: {self.total_disassembly_time} seconds\n")

            # Collect unique materials
            self.collect_unique_materials()

            # Collect unique materials for all components
            self.collect_unique_component_materials()

            # Collect unique materials for all components
            self.find_lowest_density_material()

            # Calculate the parameter total mass

            self.calculate_parameter_total_mass()

            # Calculate the parameter number of components
            self.calculate_parameter_number_of_components()

            # Calculate the parameter lifespan
            self.calculate_parameter_lifespan()

            # Calculate the parameter lifespan
            self.calculate_parameter_standardization()

            # Calculate the parameter lifespan
            self.calculate_parameter_number_of_materials()
        
            #Calculate the binary material parameters
            self.calculate_parameters_material_binary()

            # Calculate the mass share suitable for incineration with energy recovery
            self.calculate_parameter_incineration_for_energy_recovery()

            #Calculate the recycling content and recyclate content parameters
            self.calculate_recycling_and_recyclate_content()

            #Calculate the parameter origin of materials
            self.calculate_parameter_origin()

            #Calculate the parameter number of joining connections
            self.calculate_parameter_number_of_joints()

            #Calculate the parameter number of joints types
            self.calculate_parameter_number_of_joints_types()

            #Calculate the parameter used tools 
            self.calculate_parameter_used_tools()

        	#Calculate the parameter non-destructive connections
            self.calculate_parameter_joints_non_destructive()

            #Calculate the parameter ageing connections
            self.calculate_parameter_joints_ageing()

            #Calculate the parameter disassembly time
            self.calculate_parameter_disassembly_time()

            #Calculate the parameter disassembly time
            self.calculate_parameter_assembly_time()

            #Calculate the parameter material reductions
            self.calculate_development_parameters()

            #Store all parameters in a dictionary 
            self.get_all_parameters()
            current_status = self.status_bar.text()
            self.status_bar.setText(current_status + f"All parameters calculated. Total Number: {len(self.all_parameters)}\n")

            # Calculate the goal achievements
            self.calculate_goal_achievement()

            #Calculate the R-strategy scores
            self.calculate_r_strategy_scores()

            self.calculate_logistics_score()

            # Final status update
            current_status = self.status_bar.text()
            self.status_bar.setText(current_status + "Status: All calculations completed successfully!\n")

        except Exception as e:
            # Handle any errors during the calculations
            current_status = self.status_bar.text()
            self.status_bar.setText(current_status + f"Status: Error during calculations - {str(e)}\n")
            QMessageBox.critical(self, "Error", f"An error occurred during calculations:\n{str(e)}")
    
    def show_results(self):
        """
        Opens the Result Window to display the R-strategy scores in a bar chart.

        This method ensures that the R-strategy scores are calculated before opening the Result Window.
        It also collects additional information, such as total mass, volume, number of components,
        material fractions, and disassembly time, to display alongside the results.

        Args:
            None

        Returns:
            None

        Notes:
            - If the R-strategy scores are not calculated, a warning message is displayed.
            - Additional information is collected and passed to the `ResultWindow`.
            - The `ResultWindow` is opened in modal mode using `exec_()`.
        """
        # Ensure R-strategy scores are calculated
        if not hasattr(self, "r_strategy_scores") or not self.r_strategy_scores:
            QMessageBox.warning(self, "Warning", "Please calculate the R-strategy scores first.")
            return
        
        total_mass, total_volume = self.calculate_main_assembly_mass_and_volume()
        # Example usage
        material_fractions = self.calculate_material_fractions()

        #Collect additional information 
        additional_info = {
            "Total Mass [kg]": round(total_mass, 3),
            "Total Volume [mm^3]": round(total_volume, 3),
            "Number of Components": self.number_of_components,
            "Materials": material_fractions,
            "Number of Materials": len(self.material_list),
            "Number of Joining Connections": self.number_of_joining_connections,
            "Total Disassembly Time [s]": self.total_disassembly_time,

        }

        # Open the Result Window
        results_window = ResultWindow(self.r_strategy_scores, additional_info, self.goal_scores, self.data, self.all_parameters, config, self.logistics_score)
        results_window.exec_()  # Use exec_() for modal behavior

    def calculate_main_assembly_mass_and_volume(self):
        """
        Calculate the total mass and volume of the main assembly.

        This method extracts the total mass and volume of the main assembly from the hierarchical
        structure data (`self.data`). If the data is invalid or no main assembly is found, it returns 0 for both values.

        Args:
            None

        Returns:
            tuple: A tuple containing:
                - total_mass (float): The total mass of the main assembly in kilograms.
                - total_volume (float): The total volume of the main assembly in cubic millimeters.

        Notes:
            - The method assumes the main assembly is the first item in the "data" list.
            - If no valid assembly is found, the method returns (0, 0).
        """
        # Ensure self.data contains the "data" key and it's a list
        if not isinstance(self.data, dict) or "data" not in self.data or not isinstance(self.data["data"], list):
            return 0, 0

        # Find the first assembly in the list under the "data" key
        for item in self.data["data"]:
            if isinstance(item, dict) and "components" in item:
                # Extract the total mass and volume of the main assembly
                total_mass = item.get("total mass [kg]", 0)
                total_volume = item.get("volume [mm^3]", 0)  # Convert mm^3 to m^3
                return total_mass, total_volume

        # If no valid assembly is found, return 0
        return 0, 0
    
    def calculate_material_fractions(self):
        """
        Calculate the material fractions for the main assembly.

        This method traverses the hierarchical structure of the main assembly and calculates
        the percentage contribution of each material type to the total mass of the assembly.

        Args:
            None

        Returns:
            dict: A dictionary where the keys are material types and the values are their
                respective percentages of the total mass, rounded to 3 decimal places.

        Notes:
            - The method assumes the main assembly is the first item in the "data" list.
            - If no valid assembly is found, an empty dictionary is returned.
            - Material fractions are normalized to ensure they sum to 100%.
        """
        # Ensure self.data contains the "data" key and it's a list
        if not isinstance(self.data, dict) or "data" not in self.data or not isinstance(self.data["data"], list):
            return {}

        # Find the first assembly in the list under the "data" key
        first_assembly = next((item for item in self.data["data"] if isinstance(item, dict) and "components" in item), None)
        if not first_assembly:
            return {}

        # Recursive function to traverse the assembly tree
        def traverse_assembly(assembly):
            material_fractions = {}

            # Check if the current item is a leaf component (empty "components" key)
            if not assembly.get("components"):
                # Get material fractions for this component
                component_fractions = assembly.get("material_parameters", {}).get("material_fractions", [])
                component_mass = assembly.get("total mass [kg]", 0)  # Get the mass of the component
                for fraction in component_fractions:
                    material_type = fraction.get("type")
                    percentage = fraction.get("percentage", 0)
                    if material_type and component_mass > 0:
                        # Calculate the mass contribution of this material type
                        material_mass = (percentage / 100) * component_mass
                        material_fractions[material_type] = material_fractions.get(material_type, 0) + material_mass
            else:
                # Traverse subassemblies/components
                for subcomponent in assembly.get("components", []):
                    sub_fractions = traverse_assembly(subcomponent)
                    for material_type, mass in sub_fractions.items():
                        material_fractions[material_type] = material_fractions.get(material_type, 0) + mass

            return material_fractions

        # Start traversal from the main assembly
        aggregated_fractions = traverse_assembly(first_assembly)

        # Calculate the total mass
        total_mass = sum(aggregated_fractions.values())
        if total_mass > 0:
            # Normalize the material fractions to ensure they sum to 100%
            aggregated_fractions = {
                material: round((mass / total_mass) * 100, 3)  # Round to 3 decimal places
                for material, mass in aggregated_fractions.items()
            }

        return aggregated_fractions

    def calculate_classification_averages(self):
        """Derive the main assembly class and load its reference averages.

        Recompute from leaf part classifications on every run, including saved
        projects with a missing or stale assembly classification. Return False
        after one warning if no common parent or reference examples are available,
        so calculations cannot continue with missing or stale averages.
        """
        for name in (
            "average_number_of_components", "average_number_of_materials",
            "average_number_of_joining_connections", "average_number_of_joining_types",
            "average_number_of_tools", "average_disassembly_time", "average_assembly_time",
        ):
            setattr(self, name, None)

        main_classification = update_assembly_classifications(self.data, self.classification_data)
        if not main_classification:
            QMessageBox.warning(
                self, "Classification unavailable",
                "The assembly classification could not be derived from its parts. "
                "Ensure every part has a valid classification with a shared parent in the classification file.",
            )
            return False

        examples = self.find_examples_in_nested_structure(self.classification_data, main_classification)
        if not examples:
            QMessageBox.warning(
                self, "Classification examples unavailable",
                f"No reference examples are available for the derived assembly classification '{main_classification}'. "
                "Add assembly reference examples to this classification before calculating.",
            )
            return False

        # Initialize sums and counts for averaging
        total_components = 0
        total_materials = 0
        total_joining_connections = 0
        total_different_joining_connections = 0
        total_tools = 0
        total_disassembly_time = 0
        total_assembly_time = 0
        example_count = len(examples)

        # Iterate through examples and calculate sums
        for example in examples:
            total_components += example.get("number_of_components", 0)
            total_materials += example.get("number_of_materials", 0)
            total_joining_connections += example.get("number_joining_connections", 0)
            total_different_joining_connections += example.get("number_of_different_joining_connections", 0)
            total_tools += example.get("number_of_tools", 0)
            # Preserve fractional seconds in numeric or string reference times.
            try:
                total_disassembly_time += float(example.get("disassembly_time [s]", 0))
            except (TypeError, ValueError):
                total_disassembly_time += 0

            try:
                total_assembly_time += float(example.get("assembly_time [s]", 0))
            except (TypeError, ValueError):
                total_assembly_time += 0
        
        # Calculate averages and store them as instance variables
        self.average_number_of_components = total_components / example_count
        self.average_number_of_materials = total_materials / example_count
        self.average_number_of_joining_connections = total_joining_connections / example_count
        self.average_number_of_joining_types = total_different_joining_connections / example_count
        self.average_number_of_tools = total_tools / example_count
        self.average_disassembly_time = total_disassembly_time / example_count
        self.average_assembly_time = total_assembly_time / example_count
        return True

    
    def find_examples_in_nested_structure(self, classification_data, main_classification):
        """
        Recursively searches for the 'Examples' of the given main_classification in a nested classification structure.

        Args:
            classification_data (dict): The nested classification data to search within.
            main_classification (str): The classification for which to find examples.

        Returns:
            list: A list of examples associated with the given classification. Returns an empty list if no examples are found.

        Notes:
            - The method searches for a key matching the `main_classification` in the nested structure.
            - If the key is found and contains an "Examples" field, the examples are returned.
            - The search is performed recursively to handle deeply nested structures.
        """
        for key, value in classification_data.items():
            if key == main_classification and isinstance(value, dict):
                return value.get("Examples", [])
            if isinstance(value, dict):
                # Recursively search in nested dictionaries
                examples = self.find_examples_in_nested_structure(value, main_classification)
                if examples:
                    return examples
        return []  # Return an empty list if no examples are found
        
    
    def calculate_number_of_components(self):
        """
        Calculates the total number of components with an empty 'components' key in self.data.

        This method traverses the hierarchical structure in `self.data` and counts all components
        that do not have subcomponents (i.e., their 'components' key is an empty list).

        Args:
            None

        Returns:
            None

        Notes:
            - The total number of components is stored in the `self.number_of_components` attribute.
            - The method uses a recursive helper function to traverse the hierarchy.
        """
        def count_components(components):
            """Recursively count components with an empty 'components' key."""
            count = 0
            for component in components:
                if component.get("components", []) == []:  # Check if 'components' is an empty list
                    count += 1
                else:
                    # Recursively count subcomponents
                    count += count_components(component.get("components", []))
            return count

        # Start counting from the top-level assemblies
        assemblies = self.data.get("data", [])
        total_components = 0
        for assembly in assemblies:
            total_components += count_components(assembly.get("components", []))
                                                 
        self.number_of_components = total_components
    
    def calculate_number_of_material_types(self):
        """
        Calculates the total number of different material types in the assembly.

        This method traverses the hierarchical structure in `self.data` and collects all unique
        material types used in the components and subcomponents.

        Args:
            None

        Returns:
            None

        Notes:
            - The unique material types are stored in the `self.material_list` attribute.
            - The method uses a recursive helper function to traverse the hierarchy.
        """
        def collect_materials(components):
            """Recursively collect material types from all components."""
            materials = set()
            for component in components:
                # Get the material from the component
                material = component.get("material")
                if material:
                    materials.add(material)  # Add the material to the set
                # Recursively process subcomponents
                subcomponents = component.get("components", [])
                materials.update(collect_materials(subcomponents))
            return materials

        # Start collecting materials from the top-level assemblies
        assemblies = self.data.get("data", [])
        all_materials = set()
        for assembly in assemblies:
            all_materials.update(collect_materials(assembly.get("components", [])))

        # Return the unique materials
        self.material_list = list(all_materials)

        return
    
    def calculate_joining_connections(self):
        """
        Calculates the total number of joining connections in the assembly,
        taking the 'amount' into account without double-counting bidirectional connections.
        Also counts the number of unique types of joining connections, unique tools, 
        and the number of non-destructive and ageing joining connections.

        Args:
            None

        Returns:
            tuple: A tuple containing:
                - number_of_joining_connections (int): Total number of joining connections.
                - number_of_joining_types (int): Number of unique types of joining connections.
                - number_of_unique_tools (int): Number of unique tools used for joining connections.
                - number_of_non_destructive_connections (int): Number of non-destructive joining connections.
                - number_of_ageing_connections (int): Number of ageing joining connections.

        Notes:
            - The method avoids double-counting bidirectional connections by using sorted tuples.
            - Unique connection types, tools, and specific connection attributes are tracked.
            - The total disassembly time is calculated and stored as an attribute.
            - The total (re-)assembly time is calculated and stored as an attribute.
            - The method validates bidirectional connections to ensure consistency.
        """
        total_joining_connections = 0
        total_non_destructive_connections = 0
        total_ageing_connections = 0
        unique_connections = {}
        unique_connection_types = set()
        unique_tools = set()
        unique_non_destructive_connections = set()
        unique_ageing_connections = set()

        # Create a mapping of component IDs to their assigned connections
        assemblies = self.data.get("data", [])
        component_connections = {}
        for assembly in assemblies:
            self.process_components_joining(assembly.get("components", []), component_connections)

        # Iterate through the components to calculate connections and validate
        for component_name, assigned_connections in component_connections.items():
            for connected_component_name, connection_data in assigned_connections.items():
                for connection_type, connection_details in connection_data.items():
                    # Create a sorted tuple to represent the connection (to avoid double-counting)
                    connection_pair = tuple(sorted([component_name, connected_component_name]))

                    # Take the maximum amount for bidirectional connections
                    current_amount = connection_details.get("amount", 0)
                    if connection_pair not in unique_connections:
                        unique_connections[connection_pair] = current_amount
                    else:
                        unique_connections[connection_pair] = max(unique_connections[connection_pair], current_amount)

                    # Track unique connection types
                    unique_connection_types.add(connection_type)

                    # Track unique tools used for the connection
                    tools = connection_details.get("tool", "")
                    if isinstance(tools, str) and tools.strip():  # Ensure tools is a non-empty string
                        unique_tools.add(tools)

                    # Count unique non-destructive joining connections
                    total_non_destructive_connections += self.count_unique_connections_by_key(
                        connection_pair, connection_details, unique_non_destructive_connections, "non_destructive", True
                    )

                    total_ageing_connections += self.count_unique_connections_by_key(
                        connection_pair, connection_details, unique_ageing_connections, "ageing", False
                    )


                    # Validate bidirectional connection
                    self.validate_bidirectional_connection(
                        component_name, connected_component_name, connection_type, component_connections, current_amount
                    )

        # Calculate the total number of joining connections by summing the maximum amounts
        total_joining_connections = sum(unique_connections.values())

        # Compute the minimum amount of tool changes necessary
        min_tool_changes = compute_min_tool_changes(assemblies)

        # Calculate the total disassembly time
        joining_disassembly_time = self.calculate_disassembly_time(component_connections)
        total_disassembly_time = joining_disassembly_time + min_tool_changes.tool_change_count * 1.44

        # Calculate the total (re-)assembly time
        joining_assembly_times = compute_joining_assembly_time(component_connections)
        total_assembly_time = joining_assembly_times + min_tool_changes.tool_change_count * 1.44

        # Store the results in attributes
        self.number_of_joining_connections = total_joining_connections
        self.total_disassembly_time = total_disassembly_time
        self.total_assembly_time = total_assembly_time
        self.number_of_joining_types = len(unique_connection_types)
        self.number_of_unique_tools = len(unique_tools)
        self.number_of_non_destructive_connections = total_non_destructive_connections
        self.number_of_ageing_connections = total_ageing_connections

        # Return the calculated values
        return (
            self.number_of_joining_connections,
            self.number_of_joining_types,
            self.number_of_unique_tools,
            self.number_of_non_destructive_connections,
            self.number_of_ageing_connections,
        )
    
    def process_components_joining(self, components, component_connections):
        """
        Recursively processes components to calculate connections and build the mapping.

        This method traverses the hierarchical structure of components and builds a mapping
        of component names to their assigned connections.

        Args:
            components (list): List of components to process.
            component_connections (dict): Mapping of component names to their assigned connections.

        Returns:
            None

        Notes:
            - Leaf components (those without subcomponents) are added to the mapping with their assigned connections.
            - The method recursively processes subcomponents for non-leaf components.
        """
        for component in components:
            if not component.get("components", []):  # Leaf component
                component_id = component.get("name")
                assigned_connections = component.get("disassembly_parameters", {}).get("assigned_connections", {})
                component_connections[component_id] = assigned_connections
            else:  # Recursively process subcomponents
                self.process_components_joining(component.get("components", []), component_connections)


    def validate_bidirectional_connection(self, component_name, connected_component_name, connection_type, component_connections, current_amount):
        """
        Validates that the connection is bidirectional and the amounts are consistent.

        This method ensures that a connection between two components is bidirectional and that
        the amounts specified for the connection are consistent in both directions.

        Args:
            component_name (str): The name of the component initiating the connection.
            connected_component_name (str): The name of the connected component.
            connection_type (str): The type of the connection.
            component_connections (dict): A mapping of component names to their assigned connections.
            current_amount (int): The amount of the connection from the initiating component.

        Returns:
            None

        Raises:
            ValueError: If the connection is not bidirectional or if the amounts are inconsistent.

        Notes:
            - A bidirectional connection means that if component A connects to component B,
            then component B must also connect back to component A with the same connection type.
            - The method checks for the existence of the reverse connection and validates the amounts.
        """
        if connected_component_name in component_connections:
            reverse_connections = component_connections[connected_component_name]
            if component_name not in reverse_connections or connection_type not in reverse_connections[component_name]:
                raise ValueError(
                    f"Connection from {component_name} to {connected_component_name} ({connection_type}) is not bidirectional."
                )
            else:
                reverse_amount = reverse_connections[component_name][connection_type].get("amount", 0)
                if current_amount != reverse_amount:
                    raise ValueError(
                        f"Mismatch in connection amounts between {component_name} and {connected_component_name} "
                        f"({connection_type}): {current_amount} vs {reverse_amount}."
                    )
        else:
            raise ValueError(
                f"Connected component {connected_component_name} referenced by {component_name} does not exist."
            )
        
    def count_unique_connections_by_key(self, connection_pair, connection_details, unique_connections_set, key, expected_value=True):
        """
        Counts unique joining connections based on a specific key and its expected value 
        (e.g., 'non_destructive': True, 'ageing': False).

        Args:
            connection_pair (tuple): The pair of components representing the connection.
            connection_details (dict): Details of the connection.
            unique_connections_set (set): Set to track unique connections.
            key (str): The key to check in connection_details (e.g., 'non_destructive', 'ageing').
            expected_value (bool): The expected value for the key (e.g., True for 'non_destructive', False for 'ageing').

        Returns:
            int: The amount for the connection if it is unique and the key matches the expected value, otherwise 0.

        Notes:
            - A connection is considered unique if it has not been added to the `unique_connections_set`.
            - The method checks if the value of the specified key matches the `expected_value`.
            - If the connection is unique and matches the criteria, it is added to the set and its amount is returned.
        """
        if connection_details.get(key, not expected_value) == expected_value and connection_pair not in unique_connections_set:
            unique_connections_set.add(connection_pair)
            return connection_details.get("amount", 0)
        return 0

    def calculate_disassembly_time(self, component_connections):
        """
        Calculates the total disassembly time by summing the detachment time for all unique connections.

        Args:
            component_connections (dict): Mapping of component names to their assigned connections.

        Returns:
            float: The total disassembly time.

        Notes:
            - The method avoids double-counting by tracking unique connections using a set.
            - The disassembly time for each connection is calculated as the product of its amount and detachment time.
            - If no detachment time is provided for a connection, it defaults to 0.
        """
        total_disassembly_time = 0
        unique_connections = set()  # To track unique connections

        for component_name, assigned_connections in component_connections.items():
            for connected_component_name, connection_data in assigned_connections.items():
                for connection_type, connection_details in connection_data.items():
                    # Create a sorted tuple to represent the connection (to avoid double-counting)
                    connection_pair = tuple(sorted([component_name, connected_component_name, connection_type]))

                    # Ensure the connection is unique
                    if connection_pair not in unique_connections:
                        unique_connections.add(connection_pair)

                        # Calculate disassembly time for this connection
                        current_amount = connection_details.get("amount", 0)
                        detachment_time = connection_details.get("detachment_time", 0)  # Default to 0 if not provided
                        total_disassembly_time += current_amount * detachment_time
          
        return total_disassembly_time
    
    def collect_unique_materials(self):
        """
        Collects all unique materials from the examples of the main assembly's classification.

        This method retrieves the classification of the main assembly, finds examples for the classification
        in the nested structure, and collects all unique materials used in those examples.

        Args:
            None

        Returns:
            None

        Notes:
            - The method uses a set to ensure that duplicate materials are automatically handled.
            - If the main assembly does not have a classification, a warning is displayed.
            - If no examples are found for the classification, a warning is displayed.
            - The collected unique materials are stored in the `self.unique_materials` attribute.
        """
        # Load the classification structure
        classification_structure = load_json_file(self.classification_json)

        # Get the classification of the main assembly
        main_classification = self.data.get("classification")
        if not main_classification:
            QMessageBox.warning(self, "Warning", "The main assembly does not have a classification.")
            return

        # Find the examples for the main classification in the nested structure
        examples = self.find_examples_in_nested_structure(self.classification_data, main_classification)
        if not examples:
            QMessageBox.warning(self, "Warning", f"No examples found for classification '{main_classification}'.")
            return

        # Collect all materials from the examples
        unique_materials = set()  # Use a set to automatically handle duplicates
        for example in examples:
            materials = example.get("used_materials", [])
            unique_materials.update(materials)  # Add materials to the set

        # Convert the set back to a list and store it
        self.unique_materials = list(unique_materials)
    
    def collect_unique_component_materials(self):
        """
        Collects unique materials from all examples for all components in the assembly.

        This method traverses the hierarchical structure of the assembly, retrieves the classification
        for each component, and collects unique materials from the examples associated with those classifications.

        Args:
            None

        Returns:
            None

        Notes:
            - The method uses a helper function to recursively search for examples in the classification data.
            - Unique materials are stored in the `self.unique_component_materials` attribute as a list.
            - If no examples are found for a classification, the component is skipped.
        """

        # Initialize a set to store unique materials
        unique_component_materials = []

        def get_examples_for_classification(classification, classification_data):
            """
            Recursively searches for the 'Examples' key for a given classification in a nested classification structure.

            Args:
                classification (str): The classification to search for.
                classification_data (dict): The nested classification data.

            Returns:
                list: A list of examples associated with the classification, or an empty list if none are found.
            """
            # Check if the classification exists at the current level
            if classification in classification_data:
                return classification_data[classification].get("Examples", [])
            
            # Recursively search in nested dictionaries
            for key, value in classification_data.items():
                if isinstance(value, dict):  # Only recurse into dictionaries
                    examples = get_examples_for_classification(classification, value)  # Pass the correct classification_data
                    if examples:  # If examples are found, return them
                        return examples

            # Return an empty list if no examples are found
            return []

        def collect_materials_recursively(components):
            """Recursively traverse components and collect materials."""
            for component in components:
                # Check if the component is a leaf (no subcomponents)
                if not component.get("components"):  # Empty "components" key
                    classification = component.get("basic_parameters").get("classification")
                    component_name = component.get("name", "Unnamed Component")
                    if classification:
                        # Get examples for the component's classification
                        examples = get_examples_for_classification(classification, self.classification_data)
                        # Use a set to ensure unique materials for this component
                        component_materials = set()
                        for example in examples:
                            materials = example.get("used_materials", [])
                            component_materials.update(materials)  # Add materials to the set
                        # Add the unique materials for this component to the result
                        if component_materials:
                            unique_component_materials.append({
                                "component_name": component_name,
                                "materials": list(component_materials)  # Store all materials as a list
                            })
                else:
                    # Recursively process subcomponents
                    collect_materials_recursively(component.get("components", []))

        # Start the recursive collection from the top-level components of all assemblies
        assemblies = self.data.get("data", [])
        for assembly in assemblies:
            collect_materials_recursively(assembly.get("components", []))


        # Convert the set back to a list and store it
        self.unique_component_materials = list(unique_component_materials)
    
    def find_lowest_density_material(self):
        """
        Finds the material with the lowest density for each component.

        This method iterates over the unique materials for each component and identifies the material
        with the lowest density. The results are stored in the `self.lowest_density_materials` attribute.

        Args:
            None

        Returns:
            None

        Notes:
            - The method assumes that `self.unique_component_materials` contains a list of components,
            each with a "component_name" and a list of "materials".
            - The density of each material is retrieved from `self.material_data`.
            - If a material does not have a density value, it is skipped.
            - The results are stored as a list of dictionaries, each containing:
                - "component_name": The name of the component.
                - "lowest_density_material": The material with the lowest density.
                - "density": The density value of the material.
        """

        # Initialize a list to store the results
        lowest_density_materials = []

        # Iterate over each component in unique_component_materials
        for entry in self.unique_component_materials:
            component_name = entry["component_name"]
            materials = entry["materials"]

            # Find the material with the lowest density
            lowest_density = float("inf")
            lowest_density_material = None
            for material in materials:
                material_data = self.material_data.get(material)
                if material_data and "density [kg/mm^3]" in material_data:
                    density = material_data["density [kg/mm^3]"]
                    if density < lowest_density:
                        lowest_density = density
                        lowest_density_material = material

            # Store the result for the component
            if lowest_density_material:
                lowest_density_materials.append({
                    "component_name": component_name,
                    "lowest_density_material": lowest_density_material,
                    "density": lowest_density
                })

        # Store the results in self
        self.lowest_density_materials = lowest_density_materials
    
    def calculate_parameter_total_mass(self):
        """
        Calculates the quotient of the lowest mass to the current mass for each component and computes the average.

        This method traverses the hierarchical structure of the assembly, calculates the mass quotient
        (lowest mass to current mass) for each component, and computes the average of these quotients.

        Args:
            None

        Returns:
            float or None: The average mass quotient if valid quotients are found, otherwise None.

        Notes:
            - The method processes only leaf components (those with an empty "components" list).
            - The current mass is calculated using the component's volume and density.
            - The lowest mass is calculated using the component's volume and the lowest density material.
            - If the current density is lower than the lowest density, the quotient is set to 1.
            - The average of all valid mass quotients is stored in `self.parameter_total_mass`.
        """
        # Initialize a list to store the quotients
        mass_quotients = []

        def process_components(components):
            """Recursively process components and subcomponents."""
            for component in components:
                # Only process components with an empty "components" list
                if component.get("components", []) == []:  # Check if "components" is an empty list
                    component_name = component.get("name", "Unnamed Component")
                    volume = component.get("volume [mm^3]", 0)
                    current_density = component.get("density [kg/mm^3]", None)
                    current_mass = volume * current_density if current_density else None

                    # Find the lowest density material for the component
                    lowest_density_entry = next(
                        (entry for entry in self.lowest_density_materials if entry["component_name"] == component_name),
                        None
                    )
                    lowest_density = lowest_density_entry["density"] if lowest_density_entry else None
                    lowest_density_mass = volume * lowest_density if lowest_density else None

                    # Calculate the quotient of the lowest mass to the current mass
                    if current_mass and lowest_density_mass:
                        if current_density < lowest_density:
                            # If the current density is lower than the lowest density, set the quotient to 1
                            mass_quotient = 1
                        else:
                            # Otherwise, calculate the actual quotient
                            mass_quotient = lowest_density_mass / current_mass

                        mass_quotients.append(mass_quotient)

                # Recursively process subcomponents
                subcomponents = component.get("components", [])
                if subcomponents:
                    process_components(subcomponents)

        # Start processing from the top-level assemblies
        assemblies = self.data.get("data", [])
        for assembly in assemblies:
            process_components(assembly.get("components", []))

        # Calculate the average of the mass quotients
        if mass_quotients:
            self.parameter_total_mass = sum(mass_quotients) / len(mass_quotients)
        else:
            self.parameter_total_mass = None  # Handle case where no valid quotients are found
        
        return self.parameter_total_mass
    
    def calculate_parameter_number_of_components(self):
        """
        Calculates the parameter_number_of_components as the quotient of average_number_of_components 
        and the total number of components.

        Args:
            None

        Returns:
            float or None: The calculated parameter value if valid, otherwise None.

        Notes:
            - If the number of components is greater than 0, the quotient is calculated.
            - The parameter is capped at 1 if the quotient exceeds 1.
            - If no components are found (division by zero), the parameter is set to None.
        """

        if self.number_of_components> 0:
            quotient = self.average_number_of_components / self.number_of_components
            # Cap the parameter at 1 if the quotient exceeds 1
            self.parameter_number_of_components = min(quotient, 1)
        else:
            self.parameter_number_of_components = None  # Handle division by zero if no components are found
        
        return self.parameter_number_of_components
    
    def calculate_parameter_lifespan(self):
        """
        Calculates the parameter_lifespan as the quotient of the average lifespan and the maximum lifespan.

        Args:
            None

        Returns:
            float or None: The calculated parameter value if valid lifespans are found, otherwise None.

        Notes:
            - The method processes only leaf components (those with an empty "components" list).
            - Lifespan values are collected from the "basic_parameters" of each component.
            - Invalid or non-numeric lifespan values are ignored.
            - If no valid lifespans are found, the parameter is set to None.
        """
        def collect_lifespans(components):
            """Recursively collect lifespan values from all components with an empty 'components' key."""
            lifespans = []
            for component in components:
                # Only process components with an empty "components" key
                if component.get("components", []) == []:  # Check if "components" is an empty list
                    # Get the lifespan from "basic_parameters"
                    lifespan = component.get("basic_parameters", {}).get("lifespan [a]", None)
                    if lifespan is not None:
                        try:
                            # Convert lifespan to a float (or int if appropriate)
                            numeric_lifespan = float(lifespan)
                            lifespans.append(numeric_lifespan)
                        except ValueError:
                            # Handle invalid numeric values gracefully
                            print(f"Warning: Invalid lifespan value '{lifespan}' for component '{component.get('name', 'Unnamed Component')}'")
                else:
                    # Recursively process subcomponents
                    subcomponents = component.get("components", [])
                    lifespans.extend(collect_lifespans(subcomponents))
            return lifespans

        # Start collecting lifespans from the top-level assemblies
        assemblies = self.data.get("data", [])
        all_lifespans = []
        for assembly in assemblies:
            all_lifespans.extend(collect_lifespans(assembly.get("components", [])))

        # Calculate the average and maximum lifespan
        if all_lifespans:
            average_lifespan = sum(all_lifespans) / len(all_lifespans)
            max_lifespan = max(all_lifespans)
            # Calculate the parameter as the quotient of the average and maximum lifespan
            self.parameter_lifespan = average_lifespan / max_lifespan
        else:
            # Handle the case where no lifespans are found
            self.parameter_lifespan = None
        
        return self.parameter_lifespan
    
    def calculate_parameter_standardization(self):
        """
        Calculates the parameter_standardization as the quotient of standard components and the total number of components.

        Args:
            None

        Returns:
            float or None: The calculated parameter value if valid, otherwise None.

        Notes:
            - The method processes only leaf components (those with an empty "components" list).
            - A component is considered standard if its "basic_parameters" contains "standard_component" set to "Yes".
            - If no components are found (division by zero), the parameter is set to None.
        """
        def count_standard_components(components):
            """Recursively count standard components."""
            standard_count = 0
            for component in components:
                # Only process components with an empty "components" key
                if component.get("components", []) == []:  # Check if "components" is an empty list
                    # Check if the component is a standard component
                    is_standard = component.get("basic_parameters", {}).get("standard_component", "No") == "Yes"
                    if is_standard:
                        standard_count += 1
                else:
                    # Recursively process subcomponents
                    subcomponents = component.get("components", [])
                    standard_count += count_standard_components(subcomponents)
            return standard_count

        # Start counting standard components from the top-level assemblies
        assemblies = self.data.get("data", [])
        total_standard_components = 0
        for assembly in assemblies:
            total_standard_components += count_standard_components(assembly.get("components", []))

        # Calculate the standardization parameter using self.number_of_components
        if self.number_of_components > 0:
            self.parameter_standardization = total_standard_components / self.number_of_components
        else:
            self.parameter_standardization = None  # Handle division by zero if no components are found

        return self.parameter_standardization
    
    def calculate_parameter_number_of_materials(self):
        """
        Calculates the parameter_number_of_materials as the quotient of the minimum number of material groups
        (based on synergies from the classification structure) and the current number of material types.

        Args:
            None

        Returns:
            float or None: The calculated parameter value if valid, otherwise None.

        Notes:
            - The method uses a set cover algorithm to find the minimum number of material groups
            that cover all classification sets.
            - If no materials are found (division by zero), the parameter is set to None.
            - The result is stored in `self.parameter_number_of_materials`.
        """
        # Step 1: Collect currently used materials from the components
        current_materials = self.material_list
        current_number_of_material_types = len(current_materials)

        # Step 2: Find synergies using self.unique_component_materials
        # Create a set of all unique materials for each component based on the classification structure
        classification_material_sets = [set(entry["materials"]) for entry in self.unique_component_materials]

        # Step 3: Use a set cover algorithm to find the minimum number of material groups
        def find_minimum_material_groups(material_sets):
            """
            Solve the set cover problem to find the minimum number of material groups
            that cover all classification sets.
            """
            uncovered_sets = material_sets[:]
            selected_groups = []

            while uncovered_sets:
                # Find the material that covers the most uncovered sets
                material_coverage = {}
                for material_set in uncovered_sets:
                    for material in material_set:
                        if material not in material_coverage:
                            material_coverage[material] = 0
                        material_coverage[material] += 1

                # Select the material with the highest coverage
                best_material = max(material_coverage, key=material_coverage.get)

                # Add the best material to the selected groups
                selected_groups.append(best_material)

                # Remove all sets that are covered by the selected material
                uncovered_sets = [s for s in uncovered_sets if best_material not in s]

            return len(selected_groups)

        minimum_number_of_material_types = find_minimum_material_groups(classification_material_sets)

        # Step 4: Calculate the quotient
        if current_number_of_material_types > 0:
            self.parameter_number_of_materials = minimum_number_of_material_types / current_number_of_material_types
        else:
            self.parameter_number_of_materials = None  # Handle division by zero if no materials are found

        return self.parameter_number_of_materials
    
    def calculate_parameter_based_on_material_property(self, property_name, parameter_name, target_value = "false"):
        """
        Generic method to calculate a parameter based on a material property.
        The parameter is calculated as the quotient of the sum of masses with the specified property = target_value
        and the total mass of the assembly.

        Args:
            property_name (str): The name of the material property to evaluate (e.g., "recycling_criticality").
            parameter_name (str): The name of the parameter to store the result (e.g., "parameter_recycling_criticality").
            target_value (str, optional): The target value of the property to filter materials. Defaults to "false".

        Returns:
            float or None: The calculated parameter value if valid, otherwise None.

        Notes:
            - The method ensures `self.material_data` is a dictionary before proceeding.
            - The total mass of the assembly is calculated by summing the masses of all components.
            - The mass fractions of materials with the specified property matching the target value are calculated.
            - If no mass is found (division by zero), the parameter is set to None.
        """

        # Step 1:Ensure materials_data is a dictionary
        if not isinstance(self.material_data, dict):
            print("Error: materials_data is not a dictionary.")
            setattr(self, parameter_name, None)
            return None

        # Create a property map from the materials_data
        property_map = {
            material_name: str(properties.get(property_name, "Unknown")).strip().lower()
            for material_name, properties in self.material_data.items()
        }

        # Step 2: Calculate the total mass of the assembly and the mass fractions
        total_mass = 0
        material_masses = {}

        def collect_masses(components):
            """Recursively collect masses for each material."""
            nonlocal total_mass
            for component in components:
                material = component.get("material")
                mass = component.get("mass [kg]", 0)
                if material:
                    # Add the mass to the total and to the material-specific mass
                    total_mass += mass
                    if material not in material_masses:
                        material_masses[material] = 0
                    material_masses[material] += mass
                # Recursively process subcomponents
                subcomponents = component.get("components", [])
                collect_masses(subcomponents)

        # Start collecting masses from the top-level assemblies
        assemblies = self.data.get("data", [])
        for assembly in assemblies:
            collect_masses(assembly.get("components", []))

        # Step 3: Calculate the sum of masses with the specified property = target_value
        mass_with_target_value = 0
        for material, mass in material_masses.items():
            property_value = property_map.get(material, "unknown")
            if property_value == target_value:  # Compare normalized value
                mass_with_target_value += mass

        # Step 4: Calculate the quotient
        if total_mass > 0:
            setattr(self, parameter_name, mass_with_target_value / total_mass)
        else:
            setattr(self, parameter_name, None)  # Handle division by zero if no mass is found

        return getattr(self, parameter_name)
    
    def calculate_parameters_material_binary(self):
        """
        Calculates all material binary parameters (e.g., recycling_criticality, environmental_harmfulness, etc.)
        by calling the generic method for each property.

        Args:
            None

        Returns:
            None

        Notes:
            - The method defines a mapping of material properties to parameter names and their target values.
            - For each property, the `calculate_parameter_based_on_material_property` method is called.
            - The results are stored in attributes corresponding to the parameter names.
        """
        # Define the mapping of material properties to parameter names and their target values
        parameters_to_calculate = {
            "recycling_criticality": ("parameter_recycling_criticality", "false"),
            "environmental_harmfulness": ("parameter_environmental_harmfulness", "false"),
            "health_harmfulness": ("parameter_health_harmfulness", "false"),
            "additives_or_fillers": ("parameter_additives_or_fillers", "false"),
            "surface_coatings": ("parameter_surface_coatings", "false"),
            "monomaterial": ("parameter_monomaterial", "true")  # Look for "true" instead of "false"
        }

        # Loop through each property and calculate the corresponding parameter
        for property_name, (parameter_name, target_value) in parameters_to_calculate.items():
            self.calculate_parameter_based_on_material_property(
                property_name=property_name,
                parameter_name=parameter_name, 
                target_value=target_value
            )
    
    def calculate_parameter_incineration_for_energy_recovery(self):
        """Return the eligible share of the assembly's known part mass (0 to 1).

        Count each leaf occurrence once, excluding aggregate subassembly masses.
        Use a leaf's mass [kg], falling back to total mass [kg] when it is unset.
        Only positive, finite numeric masses contribute to the calculation.
        All such masses enter the denominator, including parts whose material or
        incineration flag is unknown. Only materials explicitly assessed True
        for incineration_for_energy_recovery enter the numerator.
        Return None when there is no usable part mass or materials database.
        """
        self.parameter_incineration_for_energy_recovery = None
        if not isinstance(self.material_data, dict):
            return None

        total_mass = 0.0
        eligible_mass = 0.0

        def collect_part_masses(components):
            nonlocal total_mass, eligible_mass
            for component in components:
                children = component.get("components") or []
                if children:
                    collect_part_masses(children)
                    continue

                mass = component.get("mass [kg]")
                if mass is None:
                    mass = component.get("total mass [kg]")
                if (
                    isinstance(mass, bool)
                    or not isinstance(mass, (int, float))
                    or not math.isfinite(mass)
                    or mass <= 0
                ):
                    continue

                total_mass += mass
                material = self.material_data.get(component.get("material"), {})
                if material.get("incineration_for_energy_recovery") is True:
                    eligible_mass += mass

        collect_part_masses(self.data.get("data", []))
        if total_mass > 0:
            self.parameter_incineration_for_energy_recovery = eligible_mass / total_mass
        return self.parameter_incineration_for_energy_recovery

    def calculate_recycling_and_recyclate_content(self):
        """
        Calculates the average recycling percentage and recyclate content
        for all materials in self.material_list.

        Args:
            None

        Returns:
            tuple: A tuple containing:
                - parameter_recycling_content (float or None): The average recycling percentage as a fraction (0-1),
                or None if no materials are found.
                - parameter_recyclate_content (float or None): The average recyclate content as a fraction (0-1),
                or None if no materials are found.

        Notes:
            - The method iterates through the materials in `self.material_list`.
            - Recycling percentage and recyclate content are retrieved from `self.material_data`.
            - If no materials are found, the parameters are set to None.
        """
        # Initialize sums and counts
        total_recycling_percentage = 0
        total_recyclate_content = 0
        material_count = 0

        # Iterate through the materials in self.material_list
        for material in self.material_list:
            if material in self.material_data:
                material_info = self.material_data[material]
                total_recycling_percentage += material_info.get("recycling_percentage", 0)
                total_recyclate_content += material_info.get("recyclate_content", 0)
                material_count += 1

        # Calculate averages
        if material_count > 0:
            self.parameter_recycling_content = (total_recycling_percentage / material_count)/100
            self.parameter_recyclate_content = (total_recyclate_content / material_count)/100
        else:
            self.parameter_recycling_content = None  # Handle case where no materials are found
            self.parameter_recyclate_content = None

        # Return the calculated values for further use
        return self.parameter_recycling_content, self.parameter_recyclate_content
    
    def calculate_parameter_origin(self):
        """
        Calculates the parameter_origin based on the availability of materials.
        Regionally = 1, Nationally = 0.66, Same continent = 0.33, Other continent = 0.
        The parameter is the quotient of the sum of points and the number of materials used.

        Args:
            None

        Returns:
            float or None: The calculated parameter value if valid, otherwise None.

        Notes:
            - The method assigns scores to material availability:
                - "Regionally" = 1
                - "Nationally" = 0.66
                - "Same continent" = 0.33
                - "Other continent" = 0
            - If no materials are found, the parameter is set to None.
        """
        # Initialize sums and counts
        total_points = 0
        material_count = 0

        # Define availability scoring
        availability_scores = {
            "Regionally": 1,
            "Nationally": 0.66,
            "Same continent": 0.33,
            "Other continent": 0
        }

        # Iterate through the materials in self.material_list
        for material in self.material_list:
            if material in self.material_data:
                material_info = self.material_data[material]
                availability = material_info.get("availability", "Other continent")  # Default to "Other continent"
                total_points += availability_scores.get(availability, 0)  # Default to 0 if availability is unknown
                material_count += 1

        # Calculate the parameter_origin
        if material_count > 0:
            self.parameter_origin = total_points / material_count
        else:
            self.parameter_origin = None  # Handle case where no materials are found

        # Return the calculated value for further use
        return self.parameter_origin
    
    def calculate_parameter_number_of_joints(self):
        """
        Calculates the parameter_number_of_joints as the quotient of the average number of joining connections
        and the actual number of joining connections. Caps the parameter at 1 if the quotient exceeds 1.

        Args:
            None

        Returns:
            float: The calculated parameter value.

        Notes:
            - The method uses the `calculate_disassembly_parameters` helper to perform the calculation.
            - The parameter is capped at 1 if the quotient exceeds 1.
        """
        return self.calculate_disassembly_parameters(
            self.average_number_of_joining_connections,
            self.number_of_joining_connections,
            "parameter_number_of_joints"
        )

    def calculate_parameter_number_of_joints_types(self):
        """
        Calculates the parameter_number_of_joints_types as the quotient of the average number of joining types
        and the actual number of joining types. Caps the parameter at 1 if the quotient exceeds 1.

        Args:
            None

        Returns:
            float: The calculated parameter value.

        Notes:
            - The method uses the `calculate_disassembly_parameters` helper to perform the calculation.
            - The parameter is capped at 1 if the quotient exceeds 1.
        """
        return self.calculate_disassembly_parameters(
            self.average_number_of_joining_types,
            self.number_of_joining_types,
            "parameter_number_of_joints_types"
        )

    def calculate_parameter_used_tools(self):
        """
        Calculates the parameter_used_tools as the quotient of the average number of tools
        and the actual number of unique tools used. Caps the parameter at 1 if the quotient exceeds 1.

        Args:
            None

        Returns:
            float: The calculated parameter value.

        Notes:
            - The method uses the `calculate_disassembly_parameters` helper to perform the calculation.
            - The parameter is capped at 1 if the quotient exceeds 1.
        """
        return self.calculate_disassembly_parameters(
            self.average_number_of_tools,
            self.number_of_unique_tools,
            "parameter_used_tools"
        )
    
    def calculate_parameter_assembly_time(self):
        """
        Calculates the parameter-based assembly time using the average assembly time,
        total assembly time, and the parameter key for assembly time.

        Args:
            None.

        Steps:
            1. Pass self.average_assembly_time as the average time input.
            2. Pass self.total_assembly_time as the total time input.
            3. Pass "parameter_assembly_time" as the parameter identifier.
            4. Return the result from calculate_disassembly_parameters(...).

        Returns:
            The result of calculate_disassembly_parameters(...): The calculated
            parameter assembly time value.

        Notes:
            - Uses the calculate_disassembly_parameters(...) method to perform the calculation.
            - Relies on self.average_assembly_time and self.total_assembly_time as input values.
            - Uses "parameter_assembly_time" as the lookup or calculation key.
        """
        return self.calculate_disassembly_parameters(
            self.average_assembly_time,
            self.total_assembly_time,
            "parameter_assembly_time"
        )

    def calculate_parameter_disassembly_time(self):
        """
        Calculates the parameter_used_tools as the quotient of the average number of tools
        and the actual number of unique tools used. Caps the parameter at 1 if the quotient exceeds 1.
        """
        return self.calculate_disassembly_parameters(
            self.average_disassembly_time,
            self.total_disassembly_time,
            "parameter_disassembly_time"
        )
    
    def calculate_disassembly_parameters(self, average_value, actual_value, parameter_name):
        """
        Generic method to calculate a disassembly-related parameter as the quotient of an average value
        and an actual value. Caps the parameter at 1 if the quotient exceeds 1.

        Args:
            average_value (float): The average value for the parameter (e.g., average number of tools).
            actual_value (float): The actual value for the parameter (e.g., actual number of tools used).
            parameter_name (str): The name of the parameter to store the result.

        Returns:
            float: The calculated parameter value, capped at 1, or None if actual_value is 0.
        """
        if actual_value > 0:
            quotient = average_value / actual_value
            # Cap the parameter at 1 if the quotient exceeds 1
            setattr(self, parameter_name, min(quotient, 1))
        else:
            setattr(self, parameter_name, None)  # Handle division by zero if actual_value is 0

        return getattr(self, parameter_name)
    
    def calculate_parameter_connections_attributes(self, numerator, denominator, parameter_name):
        """
        Generic method to calculate a parameter as the quotient of two attributes.
        Caps the parameter at 1 if the quotient exceeds 1.

        Args:
            numerator (float): The numerator for the calculation (e.g., number of non-destructive connections).
            denominator (float): The denominator for the calculation (e.g., total number of joining connections).
            parameter_name (str): The name of the parameter to set (e.g., 'parameter_non_destructive').

        Returns:
            float or None: The calculated parameter value, or None if the denominator is zero.
        """
        if denominator > 0:
            quotient = numerator / denominator
            setattr(self, parameter_name, min(quotient, 1))  # Cap the parameter at 1
        else:
            setattr(self, parameter_name, None)  # Handle division by zero

        return getattr(self, parameter_name)
    
    def calculate_parameter_joints_non_destructive(self):
        """
        Calculates the parameter_non_destructive as the quotient of the number of non-destructive joining connections
        and the total number of joining connections.

        Args:
            None

        Returns:
            float: The calculated parameter value.

        Notes:
            - The method uses the `calculate_parameter_connections_attributes` helper to perform the calculation.
            - The result is stored in the attribute `parameter_joints_non_destructive`.
        """
        return self.calculate_parameter_connections_attributes(
            self.number_of_non_destructive_connections,
            self.number_of_joining_connections,
            "parameter_joints_non_destructive"
        )

    def calculate_parameter_joints_ageing(self):
        """
        Calculates the parameter_ageing as the quotient of the number of ageing joining connections
        and the total number of joining connections.

        Args:
            None

        Returns:
            float: The calculated parameter value.

        Notes:
            - The method uses the `calculate_parameter_connections_attributes` helper to perform the calculation.
            - The result is stored in the attribute `parameter_joints_ageing`.
        """
        return self.calculate_parameter_connections_attributes(
            self.number_of_ageing_connections,
            self.number_of_joining_connections,
            "parameter_joints_ageing"
        )
    
    def calculate_parameter_development_by_key(self, key, parameter_name):
        """
        Calculates a parameter as the quotient of the number of components with "Yes" for a given key
        in "development_parameters" and the total number of components.

        Args:
            key (str): The key in "development_parameters" to check (e.g., "recycling_friendly_methods").
            parameter_name (str): The name of the parameter to set (e.g., "parameter_recycling_friendly_methods").

        Returns:
            float or None: The calculated parameter value, or None if the total number of components is zero.
        """
        def count_key_occurrences(components):
            """
            Recursively counts components with "Yes" for the given key in "development_parameters".
            """
            count = 0
            for component in components:
                # Check if the key is "Yes" in "development_parameters"
                if component.get("development_parameters", {}).get(key) == "Yes":
                    count += 1
                # Recursively process subcomponents
                subcomponents = component.get("components", [])
                count += count_key_occurrences(subcomponents)
            return count

        # Start counting from the top-level assemblies
        assemblies = self.data.get("data", [])
        total_count = 0
        for assembly in assemblies:
            total_count += count_key_occurrences(assembly.get("components", []))

        # Calculate the parameter
        if self.number_of_components > 0:
            parameter_value = total_count / self.number_of_components
        else:
            parameter_value = None  # Handle division by zero if no components are found

        # Dynamically set the parameter attribute
        setattr(self, parameter_name, parameter_value)
        return parameter_value
    
    def calculate_development_parameters(self):
        """
        Dynamically calculates parameters for all keys in "development_parameters".
        The parameter name is constructed as "parameter_" + key.

        Args:
            None

        Returns:
            dict: A dictionary containing the calculated parameters, where the keys are the parameter names
                (e.g., "parameter_material_reduction_low_stress") and the values are the calculated parameter values.

        Notes:
            - The method iterates through a predefined list of keys in "development_parameters".
            - For each key, the `calculate_parameter_development_by_key` method is called to calculate the parameter.
            - The results are dynamically stored as attributes of the instance.
        """
        # List of keys in "development_parameters" to calculate parameters for
        development_keys = [
            "material_reduction_low_stress",
            "recycling_friendly_methods",
            "tracking_recycling_strategies",
            "efficiency_testing_simulation",
            "robustness_long_lifespan",
            "long_term_high_quality_appearance",
            "product_material_labeling",
            "recognizability_recycling",
            "accessibility_critical_raw_materials",
            "recognizability_disposal",
            "remanufacturing_potential",
            "upcycling_potential",
        ]

        # Loop through the keys and calculate parameters dynamically
        for key in development_keys:
            parameter_name = f"parameter_{key}"  # Construct the parameter name dynamically
            self.calculate_parameter_development_by_key(key, parameter_name)

        return {
            f"parameter_{key}": getattr(self, f"parameter_{key}")
            for key in development_keys
        }
    
    def get_all_parameters(self):
        """
        Collects all calculated parameters into a single dictionary for easy access and storage.

        Returns:
            dict: A dictionary containing all calculated parameters.

        Notes:
            - General parameters, material binary parameters, and development parameters are included.
            - If a parameter is not calculated or unavailable, its value is set to `None`.
            - The collected parameters are stored in the `self.all_parameters` attribute.
        """

        # Initialize a dictionary to store all parameters
        all_parameters = {}

        # Add general parameters
        all_parameters["parameter_total_mass"] = getattr(self, "parameter_total_mass", None)
        all_parameters["parameter_number_of_components"] = getattr(self, "parameter_number_of_components", None)
        all_parameters["parameter_lifespan"] = getattr(self, "parameter_lifespan", None)
        all_parameters["parameter_standardization"] = getattr(self, "parameter_standardization", None)
        all_parameters["parameter_number_of_materials"] = getattr(self, "parameter_number_of_materials", None)
        all_parameters["parameter_recycling_content"] = getattr(self, "parameter_recycling_content", None)
        all_parameters["parameter_recyclate_content"] = getattr(self, "parameter_recyclate_content", None)
        all_parameters["parameter_origin"] = getattr(self, "parameter_origin", None)
        all_parameters["parameter_number_of_joints"] = getattr(self, "parameter_number_of_joints", None)
        all_parameters["parameter_number_of_joints_types"] = getattr(self, "parameter_number_of_joints_types", None)
        all_parameters["parameter_used_tools"] = getattr(self, "parameter_used_tools", None)
        all_parameters["parameter_disassembly_time"] = getattr(self, "parameter_disassembly_time", None)
        all_parameters["parameter_assembly_time"] = getattr(self, "parameter_assembly_time", None)
        all_parameters["parameter_joints_non_destructive"] = getattr(self, "parameter_joints_non_destructive", None)
        all_parameters["parameter_joints_ageing"] = getattr(self, "parameter_joints_ageing", None)

        # Add material binary parameters
        material_binary_keys = [
            "parameter_recycling_criticality",
            "parameter_incineration_for_energy_recovery",
            "parameter_environmental_harmfulness",
            "parameter_health_harmfulness",
            "parameter_additives_or_fillers",
            "parameter_surface_coatings",
            "parameter_monomaterial",
        ]
        for key in material_binary_keys:
            all_parameters[key] = getattr(self, key, None)

        # Add development parameters
        development_keys = [
            "material_reduction_low_stress",
            "recycling_friendly_methods",
            "tracking_recycling_strategies",
            "efficiency_testing_simulation",
            "robustness_long_lifespan",
            "long_term_high_quality_appearance",
            "product_material_labeling",
            "recognizability_recycling",
            "accessibility_critical_raw_materials",
            "recognizability_disposal",
            "remanufacturing_potential",
            "upcycling_potential",
        ]
        for key in development_keys:
            parameter_name = f"parameter_{key}"
            all_parameters[parameter_name] = getattr(self, parameter_name, None)

        self.all_parameters = all_parameters

        return self.all_parameters
 
    def calculate_goal_achievement(self):
        """
        Calculates the average parameter score for each goal defined in the JSON file.

        Returns:
            dict: A dictionary with goals as keys and their average parameter scores as values.

        Notes:
            - The method retrieves all calculated parameters using `get_all_parameters`.
            - For each goal, it calculates the average score of the associated parameters.
            - If no valid parameters are found for a goal, its score is set to `None`.
            - The results are stored in the `self.goal_scores` attribute.
        """
        # Retrieve all calculated parameters
        all_parameters = self.get_all_parameters()

        # Calculate the average score for each goal
        self.goal_scores = {}

        # Iterate through the categories and their goals
        for category, goals in self.parameter_to_goal.items():
            for goal, parameter_keys in goals.items():
                # Retrieve the parameter values for the current goal
                parameter_values = [all_parameters.get(key, None) for key in parameter_keys]

                # Filter out None values (parameters that were not calculated or are missing)
                valid_values = [value for value in parameter_values if value is not None]

                # Calculate the average score if there are valid values
                if valid_values:
                    average_score = sum(valid_values) / len(valid_values)
                else:
                    average_score = None  # No valid parameters for this goal

                # Store the average score for the goal
                self.goal_scores[goal] = average_score

        return self.goal_scores
    
    def calculate_r_strategy_scores(self):
        """
        Calculates the score for each R-strategy based on the scores of the goals and their respective weightings.

        Returns:
            dict: A dictionary with R-strategies (R0 to R9) as keys and their calculated scores as values.

        Notes:
            - The method uses the `goal_scores` attribute to retrieve the scores for each goal.
            - Weightings for each goal and R-strategy are retrieved from `self.goal_to_r_strategy`.
            - If no valid scores or weightings are found for an R-strategy, its score is set to `None`.
            - The results are stored in the `self.r_strategy_scores` attribute.
        """

        # Reject malformed rows before positional lookup can shift weights to another strategy.
        self.r_strategy_scores = {}
        validate_goal_weights(self.goal_to_r_strategy)

        # Initialize a dictionary to store the R-strategy scores
        r_strategy_scores = {f"R{i}": 0 for i in range(10)}

        # Calculate the score for each R-strategy
        for r_index in range(10):  # R0 to R9
            total_weighted_score = 0
            total_weight = 0

            for goal, weightings in self.goal_to_r_strategy.items():
                goal_score = self.goal_scores.get(goal, None)  # Get the score for the goal
                if goal_score is not None:  # Only include goals with valid scores
                    weight = weightings[r_index]  # Get the weighting for the current R-strategy
                    # Debugging line
                    total_weighted_score += goal_score * weight
                    total_weight += abs(weight)

            # Calculate the average score for the R-strategy
            if total_weight > 0:
                r_strategy_scores[f"R{r_index}"] = total_weighted_score / total_weight
            else:
                r_strategy_scores[f"R{r_index}"] = None  # Handle cases where no weightings are available
        
        self.r_strategy_scores = r_strategy_scores

        return self.r_strategy_scores
    
    def calculate_logistics_score(self):
        """
        Calculates the logistics score based on some scores of R strategies as well as the scores of some parameters and their respective weightings.

        Returns:
            float: The calculated logistics score value.

        Notes:
            - The method uses the `self.r_strategy_scores` attribute to retrieve the scores for the respective R strategy.
            - The method uses the `self.all_parameters` attribute to retrieve the scores for the respective parameters.
            - The result is stored in the `self.logistics_score` attribute.
        """

        reuse_score = self.r_strategy_scores["R3"] * 0.14
        repair_score = self.r_strategy_scores["R4"] * 0.14
        refurbish_score = self.r_strategy_scores["R5"] * 0.14
        remanufacturing_score = self.r_strategy_scores["R6"] * 0.12
        repurpose_score = self.r_strategy_scores["R7"] * 0.08

        recycling_criticality_score = self.all_parameters["parameter_recycling_criticality"] * 0.04
        monomaterial_score = self.all_parameters["parameter_monomaterial"] * 0.04
        surface_coatings_score = self.all_parameters["parameter_surface_coatings"] * 0.03
        additives_fillers_score = self.all_parameters["parameter_additives_or_fillers"] * 0.03
        recycling_content_score = self.all_parameters["parameter_recycling_content"] * 0.02
        recyclate_content_score = self.all_parameters["parameter_recyclate_content"] * 0.02
        product_material_labeling_score = self.all_parameters["parameter_product_material_labeling"] * 0.05
        tracking_recycling_strategies_score = self.all_parameters["parameter_tracking_recycling_strategies"] * 0.05
        recognizability_recycling_score = self.all_parameters["parameter_recognizability_recycling"] * 0.03
        recognizability_disposal_score = self.all_parameters["parameter_recognizability_disposal"] * 0.03
        origin_score = self.all_parameters["parameter_origin"] * 0.06

        logistics_score =  reuse_score + repair_score + refurbish_score + remanufacturing_score + repurpose_score + recycling_criticality_score + monomaterial_score + surface_coatings_score + additives_fillers_score + recycling_content_score + recyclate_content_score + product_material_labeling_score + tracking_recycling_strategies_score + recognizability_recycling_score + recognizability_disposal_score + origin_score
        self.logistics_score = logistics_score

        return self.logistics_score

class GoalDialog(QDialog):
    """
    A dialog for managing goal weightings for R-strategies.

    This class provides a user interface for viewing and editing the weightings of goals
    for each R-strategy. The weightings are displayed in a table, and users can modify
    them using spin boxes.

    Attributes:
        goal_to_r_strategy_file_path (str): The file path to the goal-to-R-strategy mapping JSON file.
        goal_data (dict): The loaded goal-to-R-strategy mapping data.
        table (QTableWidget): The table widget displaying the goals and their weightings.

    Methods:
        __init__(config: Config, parent=None): Initializes the dialog with the given configuration.
    """
    def __init__(self, config: Config, parent=None):
        """
        Initialize the GoalDialog.

        This constructor sets up the dialog's user interface, including a table for displaying
        and editing goal weightings and a save button for saving changes.

        Args:
            config (Config): An instance of the Config class containing configuration settings.
            parent (QWidget, optional): The parent widget of the dialog. Defaults to None.

        Returns:
            None
        """
        super().__init__(parent)
        self.goal_to_r_strategy_file_path = config.goal_to_r_strategy_json
        self.goal_data = load_json_file(self.goal_to_r_strategy_file_path)
        validate_goal_weights(self.goal_data)


        self.setWindowTitle("Goal Weighting")
        self.resize(800, 600)

        # Create table widget
        self.table = QTableWidget(self)
        self.table.setRowCount(len(self.goal_data))
        self.table.setColumnCount(11)  # 1 column for the goal name + 10 for weights
        self.table.setHorizontalHeaderLabels(["Goal"] + [f"R{i}" for i in range(10)])

        # Populate the table
        for row, (goal, weights) in enumerate(self.goal_data.items()):
            self.table.setItem(row, 0, QTableWidgetItem(goal))  # Goal name
            for col, weight in enumerate(weights):
                spin_box = QDoubleSpinBox(self)
                spin_box.setRange(-4.0, 4.0) 
                spin_box.setDecimals(2)  # Set the number of decimal places
                spin_box.setValue(weight)
                spin_box.valueChanged.connect(self.update_weight(row, col))
                self.table.setCellWidget(row, col + 1, spin_box)

        # Save button
        save_button = QPushButton("Save", self)
        save_button.clicked.connect(self.save_changes)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addWidget(save_button)
        self.setLayout(layout)

    def update_weight(self, row, col):
        """
        Creates a handler function to update the weight of a goal for a specific R-strategy.

        Args:
            row (int): The row index of the goal in the table.
            col (int): The column index of the R-strategy in the table.

        Returns:
            function: A handler function that updates the weight when the value changes.

        Notes:
            - The handler retrieves the goal name from `self.goal_data` using the row index.
            - The weight for the specified R-strategy is updated in `self.goal_data` when the handler is triggered.
        """
        def handler(value):
            goal = list(self.goal_data.keys())[row]
            self.goal_data[goal][col] = value
        return handler

    def save_changes(self):
        """
        Save the updated goal data back to the JSON file.

        Args:
            None

        Returns:
            None

        Notes:
            - The method writes the updated `self.goal_data` to the file specified by `self.goal_to_r_strategy_file_path`.
            - It uses the `dump_file` function to handle the file writing.
        """
        try:
            validate_goal_weights(self.goal_data)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid goal weights", str(exc))
            return
        dump_file(self.goal_data, self.goal_to_r_strategy_file_path)

class ParameterAssignmentDialog(QDialog):
    """
    A dialog for editing parameter assignments to goals.

    This class provides a user interface for assigning parameters to goals, organized by categories.
    Users can select a goal and assign or unassign parameters using checkboxes.

    Attributes:
        json_file_path (str): The file path to the parameter-to-goal mapping JSON file.
        data (dict): The loaded parameter-to-goal mapping data.
        all_parameters (set): A set of all unique parameters available for assignment.
        goal_tree (QTreeWidget): A tree widget for selecting goals organized by categories.
        checkboxes (dict): A dictionary mapping parameter names to their corresponding QCheckBox widgets.

    Methods:
        __init__(config: Config): Initializes the dialog with the given configuration.
        populate_goal_tree(): Populates the goal tree with categories and goals.
        load_goal_parameters(current, previous): Loads the parameters assigned to the selected goal.
        save_changes(): Saves the updated parameter assignments back to the JSON file.
    """
    def __init__(self, config: Config):
        """
        Initialize the ParameterAssignmentDialog.

        This constructor sets up the dialog's user interface, including a goal selector,
        parameter checkboxes, and save/close buttons.

        Args:
            config (Config): An instance of the Config class containing configuration settings.

        Returns:
            None
        """
        super().__init__()
        self.json_file_path = config.parameter_to_goal_json
        self.setWindowTitle("Edit Parameter Assignments")
        self.resize(600, 700)

        # Load JSON data
        with open(self.json_file_path, "r") as file:
            self.data = json.load(file)

        # Extract all unique parameters
        self.all_parameters = set()
        for category, goals in self.data.items():
            for goal, parameters in goals.items():
                self.all_parameters.update(parameters)

        # Create UI
        self.layout = QVBoxLayout()

        # Goal selector with categories as parent items
        self.goal_tree = QTreeWidget()
        self.goal_tree.setHeaderLabel("Goals")
        self.populate_goal_tree()
        self.goal_tree.currentItemChanged.connect(self.load_goal_parameters)
        self.goal_tree.setMinimumHeight(300) 
        self.layout.addWidget(QLabel("Select a Goal:"))
        self.layout.addWidget(self.goal_tree)

        # Checkboxes for all parameters
        self.checkbox_layout = QVBoxLayout()
        self.checkboxes = {}
        for parameter in sorted(self.all_parameters):
            checkbox = QCheckBox(parameter)
            self.checkboxes[parameter] = checkbox
            self.checkbox_layout.addWidget(checkbox)

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.checkbox_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        self.layout.addWidget(QLabel("Available Parameters:"))
        self.layout.addWidget(scroll_area)

        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_changes)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(save_button)
        button_layout.addWidget(close_button)
        self.layout.addLayout(button_layout)

        self.setLayout(self.layout)

    def populate_goal_tree(self):
        """
        Populate the QTreeWidget with categories and goals.

        This method iterates through the `self.data` dictionary, where categories are the keys
        and goals are the nested keys. It creates a tree structure in the `self.goal_tree` widget,
        with categories as top-level items and goals as child items.

        Args:
            None

        Returns:
            None
        """
        for category, goals in self.data.items():
            category_item = QTreeWidgetItem([category])
            self.goal_tree.addTopLevelItem(category_item)
            for goal in goals.keys():
                goal_item = QTreeWidgetItem([goal])
                category_item.addChild(goal_item)

    def load_goal_parameters(self, current_item):
        """
        Load parameters for the selected goal.

        This method updates the checkboxes to reflect the parameters assigned to the selected goal.

        Args:
            current_item (QTreeWidgetItem): The currently selected item in the goal tree.
            previous_item (QTreeWidgetItem): The previously selected item in the goal tree.

        Returns:
            None

        Notes:
            - If a category-level item is selected (not a goal), the method does nothing.
            - The method retrieves the category and goal from the tree structure and updates the checkboxes
            to indicate which parameters are assigned to the selected goal.
        """
        if not current_item or not current_item.parent():
            return  # Ignore category-level selections

        # Get category and goal
        category = current_item.parent().text(0)
        goal = current_item.text(0)

        # Update checkboxes
        current_parameters = set(self.data[category][goal])
        for parameter, checkbox in self.checkboxes.items():
            checkbox.setChecked(parameter in current_parameters)

    def save_changes(self):
        """
        Save changes to the JSON file.

        This method saves the updated parameter assignments for the currently selected goal
        back to the JSON file.

        Args:
            None

        Returns:
            None

        Notes:
            - If no goal is selected, a warning message is displayed.
            - The method retrieves the selected parameters from the checkboxes and updates the JSON data.
            - The updated data is saved to the file specified by `self.json_file_path`.
            - A success message is displayed upon successful saving.
        """
        current_item = self.goal_tree.currentItem()
        if not current_item or not current_item.parent():
            QMessageBox.warning(self, "Error", "No goal selected!")
            return

        # Get category and goal
        category = current_item.parent().text(0)
        goal = current_item.text(0)

        # Get selected parameters
        selected_parameters = [param for param, checkbox in self.checkboxes.items() if checkbox.isChecked()]

        # Update JSON data
        self.data[category][goal] = selected_parameters

        # Save to file
        with open(self.json_file_path, "w") as file:
            json.dump(self.data, file, indent=4)

        QMessageBox.information(self, "Success", "Changes saved successfully!")
        self.load_goal_parameters(current_item)

class ResultWindow(QDialog):
    """
    A dialog for displaying R-strategy scores in a bar chart.

    This class provides a user interface to visualize the R-strategy scores, additional information,
    and goal scores in a structured format.

    Attributes:
        r_strategy_scores (dict): A dictionary with R-strategy names (e.g., "R0", "R1") as keys
                                  and their scores as values.
        additional_info (dict): Additional information such as total mass, volume, and other metrics.
        goal_scores (dict): A dictionary with goals as keys and their average parameter scores as values.
        data (dict): The hierarchical structure data for the calculation.
        all_parameters (dict): A dictionary containing all calculated parameters.
        parameter_to_goal (dict): The mapping of parameters to goals loaded from the JSON file.

    Methods:
        __init__(r_strategy_scores, additional_info, goal_scores, data, all_parameters, config: Config, parent=None):
            Initializes the results window and sets up the user interface.
        init_ui(): Sets up the user interface for displaying the results.
    """
    def __init__(self, r_strategy_scores, additional_info, goal_scores, data, all_parameters, config: Config, logistics_score, parent=None):
        """
        Initializes the results window to display R-strategy scores in a bar chart.

        Args:
            r_strategy_scores (dict): A dictionary with R-strategy names (e.g., "R0", "R1") as keys
                                      and their scores as values.
            additional_info (dict): Additional information such as total mass, volume, and other metrics.
            goal_scores (dict): A dictionary with goals as keys and their average parameter scores as values.
            data (dict): The hierarchical structure data for the calculation.
            all_parameters (dict): A dictionary containing all calculated parameters.
            config (Config): An instance of the Config class containing configuration settings.
            logistics_score (float): A float representing the logistics score.
            parent (QWidget, optional): The parent widget of the dialog. Defaults to None.

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle("R-Strategy Results")
        self.setGeometry(100, 100, 1800, 900)

        self.parameter_to_goal_file_path = config.parameter_to_goal_json
        self.parameter_to_goal = load_json_file(self.parameter_to_goal_file_path)
        self.r_strategy_scores = r_strategy_scores
        self.additional_info = additional_info
        self.goal_scores = goal_scores
        self.data = data
        self.all_parameters = all_parameters
        self.logistics_score = logistics_score
        self.init_ui()

    def init_ui(self):
        """
        Initializes the UI for the results window with dynamically resizing figures.

        This method sets up the user interface for the `ResultWindow`, including:
        - A horizontal splitter containing:
            - Additional information about the assembly.
            - A pie chart visualizing material mass distribution.
            - An assembly tree view.
        - A vertical splitter containing:
            - A bar chart for R-strategy scores.
            - A bar chart for category averages.
            - A progess bar for the logistics score.
        - Buttons for closing the dialog and showing parameter results.

        Args:
            None

        Returns:
            None

        Notes:
            - The additional information is displayed in a scrollable area.
            - The pie chart visualizes material mass distribution.
            - The assembly tree allows users to explore the hierarchical structure.
            - The R-strategy and category average plots are dynamically generated.
        """
        # Create a main layout
        main_layout = QVBoxLayout()

        # Create a horizontal splitter for the additional information, pie chart, and assembly tree
        horizontal_splitter = QSplitter(Qt.Horizontal)

        # Left: Additional Information
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        content_widget = QWidget()
        info_layout = QVBoxLayout(content_widget)

        # Show additional information
        for key, value in self.additional_info.items():
            formatted_key = key.replace("_", " ").capitalize()

            # Skip the "Materials" key since it has the pie chart
            if key == "Materials":
                continue

            # Check if the value is a list and convert it to a comma-separated string
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            # Check if the value is a dictionary and format it as key-value pairs
            elif isinstance(value, dict):
                value = ", ".join(f"{k} ({v} %)" for k, v in value.items())

            label = QLabel(f"<b>{formatted_key}:</b> {value}")
            label.setWordWrap(True)
            info_layout.addWidget(label)

        content_widget.setLayout(info_layout)
        scroll_area.setWidget(content_widget)

        # Add the scrollable area to the horizontal splitter
        horizontal_splitter.addWidget(scroll_area)

        # Middle: Pie Chart for Materials
        materials = self.additional_info.get("Materials", {})
        if isinstance(materials, dict) and materials:
            # Create a matplotlib figure for the pie chart
            figure_pie = Figure(figsize=(5, 5))
            canvas_pie = FigureCanvas(figure_pie)
            ax = figure_pie.add_subplot(111)

            # Prepare data for the pie chart
            labels = list(materials.keys())
            sizes = list(materials.values())

            # Plot the pie chart without labels
            wedges, _ = ax.pie(sizes, startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular

            # Add a title to the pie chart
            ax.set_title("Material Mass Distribution", fontsize=14, fontweight='bold')

            # Format the legend labels to include both the material name and its value
            legend_labels = [
                f"{label} (>0.01%)" if size <= 0.1 else f"{label} ({size:.1f}%)"
                for label, size in zip(labels, sizes)
            ]

            # Add a legend closer to the pie chart
            ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(0.7, 0.5), title="Materials")

            # Add the pie chart canvas to the horizontal splitter
            horizontal_splitter.addWidget(canvas_pie)

        # Right: Assembly Tree
        tree_widget = QTreeWidget()
        tree_widget.setHeaderLabel("Assembly Tree")
        self.populate_assembly_tree(tree_widget, self.data)  # Populate the tree with data
        tree_widget.expandAll()  # Expand all items for better visibility

        # Connect the itemClicked signal to the on_tree_item_click method
        tree_widget.itemClicked.connect(self.on_tree_item_click)

        # Add the tree widget to the horizontal splitter
        horizontal_splitter.addWidget(tree_widget)
        horizontal_splitter.setMinimumHeight(300)

        # Set equal stretch factors for all three widgets in the horizontal splitter
        horizontal_splitter.setStretchFactor(0, 2)  # Additional Information
        horizontal_splitter.setStretchFactor(1, 1)  # Pie Chart
        horizontal_splitter.setStretchFactor(2, 1)  # Assembly Tree

        # Create a vertical splitter to stack the horizontal splitter and the plots
        vertical_splitter = QSplitter(Qt.Vertical)

        # Add the horizontal splitter to the vertical splitter
        vertical_splitter.addWidget(horizontal_splitter)

        # Create the first figure for R-strategy scores
        self.figure_r_strategy = Figure(figsize=(10, 5))  # Set equal height
        self.canvas_r_strategy = FigureCanvas(self.figure_r_strategy)
        self.canvas_r_strategy.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Fixed height
        self.canvas_r_strategy.setMinimumHeight(400)  # Set minimum height
        self.canvas_r_strategy.updateGeometry()

        # Adjust layout to prevent overlapping
        self.figure_r_strategy.tight_layout()

        # Add the first figure directly to the vertical splitter
        vertical_splitter.addWidget(self.canvas_r_strategy)

        # Create the second figure for category averages
        self.figure_category = Figure(figsize=(10, 5))  # Set equal height
        self.canvas_category = FigureCanvas(self.figure_category)
        self.canvas_category.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Fixed height
        self.canvas_category.setMinimumHeight(400)  # Set minimum height
        self.canvas_category.updateGeometry()

        # Adjust layout to prevent overlapping
        self.figure_category.tight_layout()

        # Add the second figure directly to the vertical splitter
        vertical_splitter.addWidget(self.canvas_category)

        self.score_card = QFrame()
        self.score_card.setObjectName("scoreCard")

        score_layout = QVBoxLayout(self.score_card)
        score_layout.setContentsMargins(12, 10, 12, 10)
        score_layout.setSpacing(6)

        score_header = QHBoxLayout()

        score_title = QLabel("Logistics score")
        score_title.setObjectName("scoreTitle")

        self.score_value_label = QLabel("0.00")
        self.score_value_label.setObjectName("scoreValue")
        self.score_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        score_header.addWidget(score_title)
        score_header.addStretch()
        score_header.addWidget(self.score_value_label)

        self.score_bar = QProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setTextVisible(False)
        self.score_bar.setFixedHeight(10)
        self.score_bar.setAccessibleName("Logistics score")

        score_hint = QLabel("Based on the WP6 study.")
        score_hint.setObjectName("scoreHint")
        score_hint.setWordWrap(True)

        score_layout.addLayout(score_header)
        score_layout.addWidget(self.score_bar)
        score_layout.addWidget(score_hint)

        self.score_card.setStyleSheet("""
            QFrame#scoreCard {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
            }

            QLabel#scoreTitle {
                font-size: 16px;
                font-weight: 600;
            }

            QLabel#scoreValue {
                font-size: 16px;
                font-weight: 700;
            }

            QLabel#scoreHint {
                color: #64748B;
            }

            QProgressBar {
                border: none;
                border-radius: 5px;
                background: #E2E8F0;
            }

            QProgressBar::chunk {
                border-radius: 5px;
                background: #87CEEB;
            }
        """)

        vertical_splitter.addWidget(self.score_card)

        # Set equal stretch factors for the horizontal splitter and the plots
        vertical_splitter.setStretchFactor(0, 1)  # Horizontal splitter
        vertical_splitter.setStretchFactor(1, 1)  # First plot (R-strategy scores)
        vertical_splitter.setStretchFactor(2, 1)  # Second plot (Category averages)
        vertical_splitter.setStretchFactor(3, 1)

        # Wrap the entire layout in a scroll area
        container_widget = QWidget()
        container_widget.setLayout(QVBoxLayout())
        container_widget.layout().addWidget(vertical_splitter)

        main_scroll_area = QScrollArea()
        main_scroll_area.setWidgetResizable(True)
        main_scroll_area.setWidget(container_widget)

        # Add the scroll area to the main layout
        main_layout.addWidget(main_scroll_area)

        # Add a horizontal layout for the Close and Show Parameter Results buttons
        button_layout = QHBoxLayout()

        # Add the Show Parameter Results button
        show_parameters_button = QPushButton("Show Parameter Results")
        show_parameters_button.clicked.connect(self.show_parameter_results)
        button_layout.addWidget(show_parameters_button)

        # Add the Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)

        

        # Add the button layout to the main layout
        main_layout.addLayout(button_layout)

        # Set the main layout for this widget
        self.setLayout(main_layout)

        # Calculate category averages
        self.category_averages = self.calculate_category_averages(self.goal_scores, self.parameter_to_goal)

        # Plot the R-strategy scores and category averages
        self.plot_r_strategy_scores(self.figure_r_strategy)
        self.plot_category_averages(self.figure_category)

        self.set_score(self.logistics_score)
    
    def closeEvent(self, event):
        """
        Handles the close event for the application.

        This method displays a message box informing the user that the calculation was successful
        and that the application will close. After the user acknowledges the message, the application quits.

        Args:
            event (QCloseEvent): The close event triggered when the window is closed.

        Returns:
            None
        """
        # Show a message box with "Calculation successful"
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Success")
        msg_box.setText("Calculation successful!\n\nThe Circular DESIGNer will close now.")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
        
        # Proceed with closing the window
        QApplication.quit()

    def plot_r_strategy_scores(self, figure):
        """
        Handles the close event for the application.

        This method displays a message box informing the user that the calculation was successful
        and that the application will close. After the user acknowledges the message, the application quits.

        Args:
            event (QCloseEvent): The close event triggered when the window is closed.

        Returns:
            None
        """
        # Extract R-strategy names and scores
        strategies = list(self.r_strategy_scores.keys())
        scores = [score * 100 if score is not None else 0 for score in self.r_strategy_scores.values()]  # Convert to percentages

        # Map R-strategy codes to their real names
        strategy_names = {
            "R0": "Refuse",
            "R1": "Rethink",
            "R2": "Reduce",
            "R3": "Reuse",
            "R4": "Repair",
            "R5": "Refurbish",
            "R6": "Remanufacture",
            "R7": "Repurpose",
            "R8": "Recycle",
            "R9": "Recover"
        }

        # Determine the best and worst scores
        max_score = max(scores)
        min_score = min(scores)

        # Assign colors: green for the best, red for the worst, and blue for others
        bar_colors = [
            "#6AA84F" if score == max_score else "#E06666" if score == min_score else "skyblue"
            for score in scores
        ]

        # Clear the figure and create a new axis
        figure.clear()
        ax = figure.add_subplot(111)

        # Plot the bars
        bars = ax.bar(strategies, scores, color=bar_colors)

        # Add labels and title
        ax.set_title("Suitability for R-Strategies", fontsize=16)
        ax.set_xlabel("R-Strategies", fontsize=12)
        ax.set_ylabel("Suitability [%]", fontsize=12)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add score labels on top of the bars
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{score:.2f}", ha='center', va='bottom', fontsize=10)

        
        # Add the first legend for R-strategy names
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=color) for color in bar_colors
        ]
        legend_labels = [f"{key} {value}" for key, value in strategy_names.items()]
        legend1 = ax.legend(
            handles=legend_handles,  # Match legend colors to bar colors
            labels=legend_labels,
            loc="lower center",  # Position the legend inside the figure, below the plot
            bbox_to_anchor=(0.5, -0.3),  # Center the legend horizontally and move it slightly down
            fontsize=8,
            ncol=10  # Make the legend horizontal
        )
        ax.add_artist(legend1)  # Add the first legend manually

        # Add the second legend for color meanings
        color_handles = [
            plt.Rectangle((0, 0), 1, 1, color="#6AA84F"),  # Green for best score
            plt.Rectangle((0, 0), 1, 1, color="#E06666")   # Red for worst score
        ]
        color_labels = [
            "Best Suitability (Green)",
            "Worst Suitability (Red)"
        ]
        legend2 = ax.legend(
            handles=color_handles,
            labels=color_labels,
            loc="upper right",  # Position the color legend in the upper right
            fontsize=8
        )

        # Adjust the layout to use all available space
        figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)  # Adjust margins for better space usage

        # Refresh the canvas
        self.canvas_r_strategy.draw()

    def plot_category_averages(self, figure):
        """
        Plots the category averages as a bar chart and ensures x-axis labels are not cut off.
        Adds scores to the bars and enables interaction to show details on bar click.

        Args:
            figure (Figure): The matplotlib figure to plot on.

        Returns:
            None

        Notes:
            - The method extracts categories and their average scores from `self.category_averages`.
            - Scores are converted to percentages, and bars are labeled with their scores.
            - A legend maps numbers on the x-axis to category names.
            - Clicking on a bar triggers the `on_bar_click` method to show details.
        """
        # Extract categories and their average scores
        categories = list(self.category_averages.keys())
        averages = [score * 100 if score is not None else 0 for score in self.category_averages.values()]  # Convert to percentages

        # Create a new axis for the figure
        ax = figure.add_subplot(111)
        ax.clear()
        bars = ax.bar(range(1, len(categories) + 1), averages, color='lightgreen', picker=True)  # Use numbers on x-axis

        # Add labels and title
        ax.set_title("Stage Averages (Click the bar for details!)", fontsize=16)
        ax.set_xlabel("Product Lifecycle Stages", fontsize=12)
        ax.set_ylabel("Average Score [%]", fontsize=12)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Replace x-axis labels with numbers
        ax.set_xticks(range(1, len(categories) + 1))
        ax.set_xticklabels(range(1, len(categories) + 1), fontsize=10)

        # Add score labels on top of the bars
        for bar, score in zip(bars, averages):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{score:.1f}%", ha='center', va='bottom', fontsize=10)

        # Add a legend mapping numbers to category names
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color='lightgreen') for _ in categories
        ]
        legend_labels = [f"{i + 1}: {category}" for i, category in enumerate(categories)]
        ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.3),
            fontsize=8,
            ncol=6 
        )
        
        # Adjust the layout to use all available space
        figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)  # Adjust margins for better space usage

        # Connect the on_bar_click method to the pick event
        figure.canvas.mpl_connect("pick_event", lambda event: self.on_bar_click(event, categories, bars))

        # Refresh the canvas
        self.canvas_category.draw()

    def set_score(self, score: float) -> None:
        """
        Updates the displayed logistics score, progress bar, tooltip, and
        accessibility description for the results window.

        Args:
            score (float): The logistics score to display. Intended values range
                from 0.0 to 1.0. Values outside this range are clamped.

        Steps:
            1. Converts the provided score to a float.
            2. Uses 0.0 when the value cannot be converted to a float.
            3. Clamps the score to the inclusive range from 0.0 to 1.0.
            4. Updates the score label with the value formatted to two decimals.
            5. Updates the progress bar using a percentage value from 0 to 100.
            6. Updates the tooltip and accessibility description with the score.

        Returns:
            None: This method updates existing user-interface widgets and does not
                return a value.

        Notes:
            - Requires `self.score_value_label` to be a QLabel-compatible widget.
            - Requires `self.score_bar` to be a QProgressBar-compatible widget.
            - Invalid values such as None or non-numeric strings are displayed as
            0.00 rather than raising an exception.
            - Scores below 0.0 display as 0.00; scores above 1.0 display as 1.00.
        """
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        score = max(0.0, min(1.0, score))

        self.score_value_label.setText(f"{score:.2f}")
        self.score_bar.setValue(round(score * 100))
        self.score_bar.setToolTip(f"Logistics score: {score:.2f} out of 1.00")
        self.score_bar.setAccessibleDescription(
            f"Logistics score {score:.2f} out of 1.00. "
            "Based on the WP6 study."
        )

    def on_bar_click(self, event, categories, bars):
        """
        Handles the click event on a bar to show the assigned goals and their respective scores in a table.

        Args:
            event: The matplotlib pick event.
            categories (list): The list of category names.
            bars (list): The list of bar objects.

        Returns:
            None

        Notes:
            - When a bar is clicked, the method identifies the corresponding category.
            - It retrieves the goals and their scores for the selected category.
            - A dialog (`CategoryDetailsDialog`) is opened to display the details.
        """
        # Check if the event is on a bar
        if event.artist in bars:
            # Get the index of the clicked bar
            index = bars.index(event.artist)
            category = categories[index]

            # Retrieve the goals and their scores for the clicked category
            goals = self.parameter_to_goal.get(category, {})
            goal_scores = {goal: self.goal_scores.get(goal, "N/A") for goal in goals.keys()}

            # Open the dialog
            dialog = CategoryDetailsDialog(category, goal_scores, self)
            dialog.exec_()
            
    def calculate_category_averages(self, goal_scores, parameter_to_goal):
        """
        Calculates the average goal scores for each category.

        Args:
            goal_scores (dict): A dictionary with goals as keys and their average scores as values.
            parameter_to_goal (dict): The parameter-to-goal mapping with categories.

        Returns:
            dict: A dictionary with categories as keys and their average scores as values.

        Notes:
            - The method iterates through each category in `parameter_to_goal`.
            - For each category, it retrieves the scores of all associated goals.
            - Goals with no valid scores (None) are excluded from the calculation.
            - If no valid scores are found for a category, its average is set to `None`.
        """
        category_averages = {}

        for category, goals in parameter_to_goal.items():
            # Retrieve the scores for all goals in the category
            scores = [goal_scores.get(goal, None) for goal in goals.keys()]

            # Filter out None values (goals with no valid scores)
            valid_scores = [score for score in scores if score is not None]

            # Calculate the average score for the category
            if valid_scores:
                category_averages[category] = sum(valid_scores) / len(valid_scores)
            else:
                category_averages[category] = None  # No valid scores for this category

        return category_averages

    def populate_assembly_tree(self, tree_widget, data):
        """
        Populates the assembly tree with assemblies and their respective components.

        Args:
            tree_widget (QTreeWidget): The tree widget to populate.
            data (dict): The data containing assemblies and components.

        Returns:
            None

        Notes:
            - The method retrieves assemblies from the `data` dictionary.
            - Each assembly is added as a top-level item in the tree.
            - Components of each assembly are added as child items recursively.
        """
        assemblies = data.get("data", [])

        for assembly in assemblies:
            # Create a top-level item for the assembly
            assembly_name = assembly.get("name", "Unnamed Assembly")
            assembly_item = QTreeWidgetItem([assembly_name])
            tree_widget.addTopLevelItem(assembly_item)

            # Add components as child items
            components = assembly.get("components", [])
            self.add_components_to_tree(assembly_item, components)

    def add_components_to_tree(self, parent_item, components):
        """
        Recursively adds components and their subcomponents to the tree.

        Args:
            parent_item (QTreeWidgetItem): The parent tree item.
            components (list): The list of components to add.

        Returns:
            None

        Notes:
            - Each component is added as a child item to the parent tree item.
            - The full component data is stored in the item's data for later use.
            - Subcomponents are recursively added as child items.
        """
        for component in components:
            # Create a tree item for the component
            component_name = component.get("name", "Unnamed Component")
            component_item = QTreeWidgetItem([component_name])

            # Store the full component data in the item's data
            component_item.setData(0, Qt.UserRole, component)

            parent_item.addChild(component_item)

            # Recursively add subcomponents (if any)
            subcomponents = component.get("components", [])
            if subcomponents:
                self.add_components_to_tree(component_item, subcomponents)

    def on_tree_item_click(self, item, column):
        """
        Handles the click event on a tree item to show the component's data in a dialog.

        Args:
            item (QTreeWidgetItem): The clicked tree item.
            column (int): The column of the clicked item.

        Returns:
            None

        Notes:
            - The method retrieves the component data stored in the item's data.
            - If the component data exists, it opens a `ComponentDetailsDialog` to display the details.
        """
        # Retrieve the component data stored in the item's data
        component_data = item.data(0, Qt.UserRole)

        if component_data:
            # Extract the "name" field from the component data
            component_name = component_data.get("name", "Unnamed Component")

            # Open a dialog to show the component's data
            dialog = ComponentDetailsDialog(component_name, component_data, self)
            dialog.exec_()
    
    def show_parameter_results(self):
        """
        Opens the Parameter Results dialog to display all parameters.

        Args:
            None

        Returns:
            None

        Notes:
            - The method creates an instance of `ParameterResultsDialog`.
            - It passes `self.all_parameters` to the dialog for display.
            - The dialog is executed modally using `exec_()`.
        """
        dialog = ParameterResultsDialog(self.all_parameters, self)
        dialog.exec_()

class ComponentDetailsDialog(QDialog):
    """
    A dialog for displaying detailed information about a specific component.

    Attributes:
        component_name (str): The name of the component.
        component_data (dict): The data of the component.

    Methods:
        __init__(component_name, component_data, parent=None): Initializes the dialog with the component details.
        add_data_to_layout(data, layout): Adds the component data to the provided layout.
    """
    def __init__(self, component_name, component_data, parent=None):
        """
        Initializes the dialog to display component details.

        Args:
            component_name (str): The name of the component.
            component_data (dict): The data of the component.
            parent: The parent widget (optional).

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle(f"Details for Component: {component_name}")
        self.setMinimumSize(500, 400)  # Set a reasonable minimum size

        # Create a main layout for the dialog
        main_layout = QVBoxLayout(self)

        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Create a container widget for the scroll area
        container_widget = QWidget()
        scroll_layout = QVBoxLayout(container_widget)

        # Add the component data to the layout
        self.add_data_to_layout(component_data, scroll_layout)

        # Set the container widget as the scroll area's widget
        scroll_area.setWidget(container_widget)

        # Add the scroll area to the main layout
        main_layout.addWidget(scroll_area)

        # Add a close button
        close_button = QPushButton("Close")

    def add_data_to_layout(self, data, layout, indent=0):
        """
        Recursively adds data as labels to the layout with increased indentation
        and different font sizes for nested levels, following a typical order.

        Args:
            data (dict, list, or other): The data to display.
            layout (QVBoxLayout): The layout to add the labels to.
            indent (int): The indentation level for nested data.

        Returns:
            None

        Notes:
            - Keys are displayed in a typical order (e.g., "name", "material", etc.) before others.
            - Nested data is indented, and font size decreases with depth.
            - Supports rich text formatting for labels.
            """
        # Define the typical order of keys
        order = [
            "name", "material", "mass [kg]", "volume [mm^3]",
            "density [kg/mm^3]", "basic_parameters", "disassembly_parameters",
            "material_parameters", "development_parameters"
        ]

        indent_str = "&nbsp;" * (indent * 8)  # Use more spaces for deeper indentation

        if isinstance(data, dict):
            # Process keys in the typical order first
            for key in order:
                if key in data:
                    value = data[key]

                    # Skip specific keys
                    if key in ["id", "assessed"]:
                        continue

                    # Format the key with increased font size for higher levels
                    font_size = max(14 - indent, 11)  # Decrease font size with depth, minimum 10
                    formatted_key = f'<span style="font-size:{font_size}px;">{indent_str}<b>{key.replace("_", " ").capitalize()}:</b></span>'

                    # Add the key as a label
                    label = QLabel(formatted_key)
                    label.setWordWrap(True)
                    label.setTextFormat(Qt.RichText)  # Enable rich text interpretation
                    layout.addWidget(label)

                    # Recursively handle nested data
                    self.add_data_to_layout(value, layout, indent + 1)

            # Process remaining keys that are not in the typical order
            for key, value in data.items():
                if key not in order and key not in ["id", "assessed"]:
                    # Format the key with increased font size for higher levels
                    font_size = max(14 - indent, 10)
                    formatted_key = f'<span style="font-size:{font_size}px;">{indent_str}<b>{key.replace("_", " ").capitalize()}:</b></span>'

                    # Add the key as a label
                    label = QLabel(formatted_key)
                    label.setWordWrap(True)
                    label.setTextFormat(Qt.RichText)
                    layout.addWidget(label)

                    # Recursively handle nested data
                    self.add_data_to_layout(value, layout, indent + 1)

        elif isinstance(data, list):
            for index, item in enumerate(data):
                # Add a label for each list item
                if isinstance(item, (dict, list)):
                    # Handle nested dictionaries or lists
                    self.add_data_to_layout(item, layout, indent + 1)
                else:
                    # Add simple list items with indentation
                    font_size = max(14 - indent, 10)  # Decrease font size with depth, minimum 10
                    formatted_item = f'<span style="font-size:{font_size}px;">{indent_str}- {item}</span>'
                    label = QLabel(formatted_item)
                    label.setWordWrap(True)
                    label.setTextFormat(Qt.RichText)  # Enable rich text interpretation
                    layout.addWidget(label)

        else:
            # Handle simple key-value pairs or standalone values
            font_size = max(14 - indent, 10)  # Decrease font size with depth, minimum 10
            formatted_value = f'<span style="font-size:{font_size}px;">{indent_str}{data}</span>'
            label = QLabel(formatted_value)
            label.setWordWrap(True)
            label.setTextFormat(Qt.RichText)  # Enable rich text interpretation
            layout.addWidget(label)

class CategoryDetailsDialog(QDialog):
    """
    A dialog for displaying detailed information about a specific category.

    Attributes:
        category (str): The name of the category.
        goal_scores (dict): A dictionary of goals and their scores.

    Methods:
        __init__(category, goal_scores, parent=None): Initializes the dialog with the category details.
    """
    def __init__(self, category, goal_scores, parent=None):
        """
        Initializes the dialog to display category details.

        Args:
            category (str): The name of the category.
            goal_scores (dict): A dictionary of goals and their scores.
            parent: The parent widget (optional).

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle(f"Details for Category: {category}")
        self.setMinimumSize(500, 300)

        # Create a layout for the dialog
        layout = QVBoxLayout(self)

        # Add a label for the category
        category_label = QLabel(f"<b>Stage:</b> {category}")
        category_label.setStyleSheet("""
            font-size: 16px;  /* Larger font size */
            font-weight: bold;  /* Bold font */
            margin-bottom: 10px;  /* Add spacing below the label */
        """)
        layout.addWidget(category_label)

        # Add goal-score pairs as labels
        for goal, score in goal_scores.items():
            # Format the score
            score_text = f"{score * 100:.2f}%" if isinstance(score, (int, float)) else "N/A"
            # Create a label for the goal and score
            goal_label = QLabel(f"<b>{goal}:</b> {score_text}")
            goal_label.setWordWrap(True)  # Enable word wrapping for long text
            layout.addWidget(goal_label)

        # Add a close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        # Set the layout
        self.setLayout(layout)

class ParameterResultsDialog(QDialog):
    """
    A dialog for displaying all calculated parameters and their values.

    Attributes:
        all_parameters (dict): A dictionary of parameters to display.

    Methods:
        __init__(all_parameters, parent=None): Initializes the dialog with the provided parameters.
    """
    def __init__(self, all_parameters, parent=None):
        """
        Initializes the Parameter Results dialog.

        Args:
            all_parameters (dict): A dictionary of parameters to display.
            parent: The parent widget (optional).

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle("Parameter Results")
        self.setGeometry(200, 200, 600, 400)

        # Create a scrollable area for the parameters
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        # Create a widget to hold the parameter labels
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        # Add each parameter as a label-value pair
        for key, value in all_parameters.items():
            formatted_key = key.replace("_", " ").capitalize()
            formatted_value = round(value, 2) if isinstance(value, (int, float)) else "N/A"
            label = QLabel(f"<b>{formatted_key}:</b> {formatted_value}")
            label.setWordWrap(True)
            layout.addWidget(label)

        content_widget.setLayout(layout)
        scroll_area.setWidget(content_widget)

        # Create a layout for the dialog
        dialog_layout = QVBoxLayout(self)
        dialog_layout.addWidget(scroll_area)

        # Add a Close button to the dialog
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        dialog_layout.addWidget(close_button)

        # Set the layout for the dialog
        self.setLayout(dialog_layout)
    
    
if __name__ == "__main__":
    """
    Entry point for the application.

    This block initializes the application, sets up the main window, and starts the event loop.

    Steps:
        1. Enables High DPI scaling for better display on high-resolution screens.
        2. Creates a `QApplication` instance.
        3. Loads the configuration using the `Config` class.
        4. Sets the application icon.
        5. Creates and displays the main window (`MainWindow`).
        6. Applies dynamic stylesheets to the application.
        7. Starts the application's event loop.

    Returns:
        None
    """
    # Windows taskbar identity: a stable, unique identifier needed.
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "ZEvRA.CircularDesigner.1.0"
        )

    # Enable High DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)

    try:
        import pyi_splash
        pyi_splash.update_text("Opening application...")
    except Exception:
        pyi_splash = None

    config = Config()
    
    icon_path = str(BASE_DIR / "tool_logo.png")

    app.setWindowIcon(QIcon(icon_path))
    window = MainWindow(config)
    window.setWindowIcon(QIcon(icon_path))
    window.show() 

    if pyi_splash:
        pyi_splash.close()
    
    apply_dynamic_stylesheet(app, window) 

    sys.exit(app.exec_())  # Starts the application's event loop
