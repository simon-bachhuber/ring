from pathlib import Path

import jax
from jax import random
import numpy as np
from PyQt6 import QtWidgets
from vispy.app import use_app

from x_xy import base
from x_xy import forward_kinematics
from x_xy import load_example
from x_xy import load_sys_from_xml
from x_xy.algorithms.generator import _setup_fn_randomize_positions
from x_xy.experimental import pipeline
from x_xy.io import list_examples
from x_xy.render.render import VispyScene
from x_xy.subpkgs import exp_data
from x_xy.subpkgs import sys_composer

forward_kinematics = jax.jit(forward_kinematics)

EXAMPLE_CHOICES = list_examples()
EXPERIMENT_CHOICES = exp_data.list_experiments()

CANVAS_SIZE = (800, 720)  # (width, height)


def to_int(value: str):
    try:
        return int(value)
    except:  # noqa: E722
        return value.strip().lstrip('"').rstrip('"')


class MyMainWindow(QtWidgets.QMainWindow):
    _sys = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout()

        self._controls = Controls()
        main_layout.addWidget(self._controls)
        self._canvas_wrapper = VispyScene(size=CANVAS_SIZE, bgcolor="black")
        self._canvas_wrapper.disable_xyz_tranform1()
        main_layout.addWidget(self._canvas_wrapper.canvas.native)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self._connect_controls()

    def _connect_controls(self):
        self._controls.refresh_button.clicked.connect(self.set_sys)
        self._controls.joint_control_slider.valueChanged.connect(self.set_joint_control)

    def set_sys(self):
        sys_id = self._controls.file_dialog_selected_file.text()
        if sys_id in EXAMPLE_CHOICES:
            sys = load_example(sys_id)
        elif sys_id in EXPERIMENT_CHOICES:
            sys = exp_data.load_sys(sys_id)
        else:
            sys = load_sys_from_xml(sys_id)

        dialog_moprh = self._controls.dialog_morph.text()
        if dialog_moprh != "":
            new_structure = [to_int(name) for name in dialog_moprh.split(",")]
            sys = sys_composer.morph_system(sys, new_structure)

        dialog_seed = self._controls.dialog_seed.text()
        if dialog_seed != "":
            seed = int(dialog_seed)
            key = random.PRNGKey(seed)
            c1, c2, c3 = random.split(key, 3)
            sys = pipeline.rr_joint.setup_fn_randomize_joint_axes(c1, sys)
            sys = _setup_fn_randomize_positions(c2, sys)
            # sys = pipeline.generator._setup_fn_randomize_transform1_rot(c3, sys, 0.2)

        self._controls.joint_control.clear()
        self._controls.joint_control.addItems(sys.link_names)

        self._sys = sys

        state = base.State.create(sys)

        _, state = forward_kinematics(sys, state)
        self._canvas_wrapper.init(sys.geoms)
        self._canvas_wrapper.update(state.x)

    def set_joint_control(self, slider_value):
        if self._sys is not None:
            q_link_name = slider_value / 100
            link_name = self._controls.joint_control.currentText()
            state = base.State.create(self._sys)
            q = np.array(state.q)
            q[self._sys.idx_map("q")[link_name]] = q_link_name
            _, state = forward_kinematics(self._sys, state.replace(q=q))
            self._canvas_wrapper.update(state.x)

    def set_qvalue(self):
        pass

    def toggle_cs(self):
        pass


class Controls(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout()
        self.sys_chooser = QtWidgets.QComboBox()
        self.sys_chooser.addItems(EXAMPLE_CHOICES + EXPERIMENT_CHOICES)
        self.sys_chooser.currentTextChanged.connect(self.update_select_system)
        layout.addWidget(self.sys_chooser)

        self.file_dialog_button = QtWidgets.QPushButton("Select Path")
        self.file_dialog_button.clicked.connect(self.open_file_dialog)

        layout.addWidget(QtWidgets.QLabel("Currently selected system:"))
        self.file_dialog_selected_file = QtWidgets.QLineEdit()
        layout.addWidget(self.file_dialog_button)
        layout.addWidget(self.file_dialog_selected_file)

        layout.addWidget(QtWidgets.QLabel("Morphology"))
        self.dialog_morph = QtWidgets.QLineEdit("")
        layout.addWidget(self.dialog_morph)

        layout.addWidget(QtWidgets.QLabel("Seed"))
        self.dialog_seed = QtWidgets.QLineEdit("")
        layout.addWidget(self.dialog_seed)

        self.refresh_button = QtWidgets.QPushButton("Refresh / Reload")
        layout.addWidget(self.refresh_button)

        layout.addWidget(QtWidgets.QLabel("Joint"))
        self.joint_control = QtWidgets.QComboBox()
        layout.addWidget(self.joint_control)

        layout.addWidget(QtWidgets.QLabel("Joint Value"))
        self.joint_control_slider = QtWidgets.QSlider()
        self.joint_control_slider.setMinimum(-500)
        self.joint_control_slider.setMaximum(500)
        layout.addWidget(self.joint_control_slider)

        layout.addStretch(1)
        self.setLayout(layout)

    def open_file_dialog(self):
        filename, ok = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a system", filter="*.xml"
        )

        if filename is not None:
            self.file_dialog_selected_file.setText(str(Path(filename)))

    def update_select_system(self, sys_id):
        self.file_dialog_selected_file.setText(sys_id)


if __name__ == "__main__":
    app = use_app("pyqt6")
    app.create()
    win = MyMainWindow()
    win.show()
    app.run()
