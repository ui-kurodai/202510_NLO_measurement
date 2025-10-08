from PyQt6.QtWidgets import (
    QGroupBox, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QComboBox, QMessageBox, QFormLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import pyvisa
from devices.boxcar_control import BoxcarInterfaceController
import logging
import time
import numpy as np
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - in %(filename)s - %(message)s')


class BoxcarWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("SR245 Boxcar Control", parent)

        self.controller = None
        self.polling_thread = None

        # UI Element
        # GPIB port selection
        self.scan_resource_btn = QPushButton("Scan GPIB resource")
        self.scan_resource_btn.clicked.connect(self.scan_gpib_resources)
        self.resource_combo = QComboBox()

        # establish connection
        self.connect_btn = QPushButton("Open Port")
        self.connect_btn.clicked.connect(self.toggle_connection)

        # display analog port output, default port == 1,2
        self.analog_output = [[1, np.nan],
                              [2, np.nan]]
        
        self.add_port_selector = QComboBox()
        self.add_port_selector.addItems([str(i+1) for i in range(8)])

        self.add_port_btn = QPushButton("+")
        self.add_port_btn.setEnabled(False)
        self.add_port_btn.clicked.connect(lambda: self.handle_port("+"))
        
        # Layout
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.addWidget(self.scan_resource_btn)
        self.layout.addWidget(self.resource_combo)
        self.layout.addWidget(self.connect_btn)

        # analog voltage output
        # somehow needs update upon each event
        self.output_layout_all = QVBoxLayout()
        self.output_layout_all.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.port_labels = {}
        self.del_port_btns = {}
        self.port_rows = {}
        for port, v in self.analog_output:
            # add label
            if v is np.nan:
                label = QLabel(f"Port {port}: ---- mV")
            else:
                label = QLabel(f"Port {port}: {v:.3f} mV")
            self.port_labels[port] = label
            
            # add del_btn
            del_port_btn = QPushButton("-")
            del_port_btn.setEnabled(False)
            del_port_btn.clicked.connect(lambda _, p=port: self.handle_port("-", p))
            self.del_port_btns[port] = del_port_btn

            # add row
            output_layout = QHBoxLayout()
            output_layout.addWidget(self.del_port_btns[port])
            output_layout.addWidget(self.port_labels[port])
            self.port_rows[port] = output_layout

            self.output_layout_all.addLayout(output_layout)
        self.layout.addLayout(self.output_layout_all)

        add_output_layout = QHBoxLayout()
        add_output_layout.addWidget(self.add_port_btn)
        add_output_layout.addWidget(self.add_port_selector)
        self.layout.addLayout(add_output_layout)

        self.setLayout(self.layout)

    def scan_gpib_resources(self):
        self.resource_combo.clear()
        rm = pyvisa.ResourceManager()
        for res in rm.list_resources():
            label = res
            try:
                with rm.open_resource(res) as inst:
                    inst.timeout = 1000
                    idn = inst.resource_manufacturer_name
                    label = f"{res} | {idn.strip()}"
            except Exception as e:
                logging.warning(f"Failed to query IDN for {res}: {e}")
            finally:
                try:
                    inst.close()
                except: pass
                self.resource_combo.addItem(label, userData=res)

    def get_active_ports(self):
        return [p for p, _ in self.analog_output]

    def toggle_connection(self):
        # connection on
        if self.controller is None:
            resource = self.resource_combo.currentData()
            if resource == "":
                QMessageBox.warning(self, "Device Not Found", "No GPIB resource is selected.")
                return
            try:
                self.controller = BoxcarInterfaceController()
                self.controller.connect(resource)
            except Exception as e:
                QMessageBox.critical(self, "Connection Error", str(e))
            else:
                logging.info("Boxcar connected")
                self.connect_btn.setText("Close Port")
                self.set_controls_enabled(True)

                self.polling_thread = BoxcarPollingThread(
                    self.controller,
                    get_ports_func=self.get_active_ports,
                    interval=1.0
                    )
                self.polling_thread.status_updated.connect(self.update_analog_output)
                self.polling_thread.start() 

        # connection off
        else:
             # stop polling
            if self.polling_thread is not None:
                try:
                    self.polling_thread.stop()
                except Exception as e:
                    logging.error(f"Failed to stop polling: {e}")
            # close connection
            if not self.controller.is_connected:
                logging.debug("Attempted to close connection when not connected")
                return
            try:
                self.controller.disconnect()
            except Exception as e:
                logging.error(f"Failed to close connection with boxcar: {e}")

            # delete instance anyway
            self.controller = None
            self.connect_btn.setText("Open Port")
            self.polling_thread = None
            self.set_controls_enabled(False)
            self.clear_output()

    def set_controls_enabled(self, enabled):
        for btns in self.del_port_btns.values():
            btns.setEnabled(enabled)
        self.add_port_btn.setEnabled(enabled)

    def handle_port(self, operation: str, port: Optional[int] =None):
        if operation == "+":
            port = int(self.add_port_selector.currentText())
            if port == "":
                QMessageBox.information(self, "Port Not Found","No port selected")
                return
            if any(p == port for p, _ in self.analog_output):
                QMessageBox.information(self, "Duplicate", f"Port {port} already added.")
                return
            # update data
            self.analog_output.append([port, np.nan])
            # upate label
            self.port_labels[port] = QLabel(f"Port {port}: ---- mV")
            self.del_port_btns[port] = QPushButton("-")
            self.del_port_btns[port].clicked.connect(lambda _, p=port:self.handle_port("-", p))

            output_layout = QHBoxLayout()
            output_layout.addWidget(self.del_port_btns[port])
            output_layout.addWidget(self.port_labels[port])
            self.port_rows[port] = output_layout

            self.output_layout_all.addLayout(output_layout)
            
        
        elif operation == "-":
            port = port
            # update data
            if all(p != port for p, _ in self.analog_output):
                QMessageBox.information(self, "No port matched", f"Port {port} does not exist.")
                return
            new_list = [[p, _] for p, _ in self.analog_output if p != port]
            self.analog_output = new_list

            # update label
            label = self.port_labels.pop(port, None)    # delete port & return label or None(if port not matched)
            btn = self.del_port_btns.pop(port, None)
            print("row:", self.port_rows)
            row = self.port_rows.pop(port, None)
            
            if row is not None:
                # delete btn, label
                for i in reversed(range(row.count())):
                    item = row.itemAt(i)
                    print("item:", item)
                    widget = item.widget()
                    if widget is not None:
                        widget.setParent(None)
                        widget.deleteLater()

                # delete row
                self.output_layout_all.removeItem(row)
                print(self.output_layout_all)

            
    def update_analog_output(self, output_dict: dict):
        # update gui-internal value
        if output_dict is not None:
            new_output = []
            for port, output in output_dict.items():
                try:
                    new_output.append([port, output])
                except Exception as e:
                    logging.error(f"Failed to update output data: {e}")
            self.analog_output = new_output

        # update_label
        for port, v in self.analog_output:
            try:
                if np.isnan(v):
                    self.port_labels[port].setText(f"Port {port}: ---- mV")
                else:
                    self.port_labels[port].setText(f"Port {port}: {v:.3f} mV")
            except Exception as e:
                    logging.error(f"Failed to update output label: {e}")

    def clear_output(self):
        for i in range(len(self.analog_output)):
            port, _ = self.analog_output[i]
            self.analog_output[i][1] = np.nan  # overwrite
            if port in self.port_labels:
                self.port_labels[port].setText(f"Port {port}: ---- mV")


    def shutdown(self):
        if self.controller is not None:
            if self.controller.is_connected:
                try:
                    self.controller.close()
                    self.controller = None
                    logging.info("Closed ports before shutdown")
                except Exception as e:
                    logging.warning(f"Failed to disconnect: {e}")

class BoxcarPollingThread(QThread):
    status_updated = pyqtSignal(dict)

    def __init__(self, controller, get_ports_func, interval=0.2, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.get_ports = get_ports_func
        self.interval = interval
        self._running = True

    def run(self):
        while self._running:
            try:
                outputs = {}
                ports = self.get_ports()
                for port in ports:
                    outputs[port] = self.controller.read_analog(port)
                self.status_updated.emit(outputs)
            except Exception as e:
                logging.error(f"Polling failed: {e}")
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        self.wait()