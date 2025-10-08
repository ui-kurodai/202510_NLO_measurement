import pyvisa
from pyvisa.errors import VisaIOError, InvalidSession, LibraryError
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - in %(filename)s - %(message)s")


class BoxcarInterfaceController:
    def __init__(self, timeout_ms=5000) -> None:
        self._resource_manager = pyvisa.ResourceManager()
        self._timeout_ms = timeout_ms
        self.inst = None
    

    @property
    def gpib_resources_list(self) -> list[str]:
        return self._resource_manager.list_resources()
    

    @property
    def gpib_resources_dict(self) -> dict:
        """
        returns a dictionary
        - key: resource name (str)
        - value: detailed info of resource (str)
        """
        return self._resource_manager.list_resources_info()
    

    def connect(self, resource:str) -> None:
        if self.inst is None:
            try:
                daq = self._resource_manager.open_resource(resource)
                self.inst = daq
                self.inst.timeout = self._timeout_ms
            except (VisaIOError, InvalidSession, LibraryError, Exception) as e:
                logging.error(f"Failed to connect SR245: {e}")
            
    

    def autodetect_SR245(self) -> Optional[str]:
        for resource_name in self._resource_manager.list_resources():
            try:
                with self._resource_manager.open_resource(resource_name) as inst:
                    inst.timeout = self._timeout_ms
                    idn = inst.query("*IDN?")
                    if "SR245" in idn or "Stanford" in idn:
                        logging.info(f"Detected SR245: {idn.strip()} at {resource_name}")
                        return resource_name
            except Exception as e:
                logging.debug(f"Resource {resource_name} did not respond to *IDN?: {e}")
        logging.warning("SR245 not detected.")
        return None


    def disconnect(self) -> None:
        if self.inst is None:
            logging.warning("Attemped to disconnect SR245 while not connected")
        else:
            try:
                self.inst.close()
            except (VisaIOError, InvalidSession, LibraryError, Exception) as e:
                logging.error(f"Failed to disconnect SR245: {e}")
            finally:
                self.inst = None


    def __del__(self) -> None:
        if self.inst is not None:
            try:
                self.disconnect()
            except Exception:
                pass
        

    @property
    def is_connected(self) -> bool:
        return self.inst is not None

    
    def send_command(self, command:str) -> None:
        if self.is_connected:
            try:
                self.inst.write(command)
            except (VisaIOError, InvalidSession, LibraryError, Exception) as e:
                logging.error(f"Failed to send command {command} to SR245: {e}")
    

    def read_response(self) -> Optional[str]:
        if self.is_connected:
            try:
                return self.inst.read()
            except (VisaIOError, InvalidSession, LibraryError, Exception) as e:
                logging.error(f"Failed to read response from SR245: {e}")
                return None
        else:
            return None
    

    @property
    def idn(self) -> Optional[str]:
        if self.is_connected:
            try:
                self.send_command("*IDN?")
                return self.read_response()
            except (VisaIOError, InvalidSession, LibraryError, Exception) as e:
                logging.error(f"Failed to get IDN: {e}")
        return None
            

    def set_input(self, number_of_ports:int) -> None:
        """
        Designates the first n analog ports as inputs, the remainder become outputs.
        Command: I<n> n=0~8
        """
        if not (0 <= number_of_ports <= 8):
            logging.warning("Invalid number of ports")
            return
        command = f"I{number_of_ports}"
        self.send_command(command)
    

    def read_analog(self, port_number:int) -> Optional[float]:
        """
        Returns the value of the designatedanalog port.
        Command: ?<n> n=1~8
        """
        if not (1 <= port_number <= 8):
            logging.warning(f"Invalid port number: {port_number}")
            return None
        command = f"?{port_number}"
        self.send_command(command)
        response = self.read_response()
        try:
            return float(response)
        except (TypeError, Exception) as e:
            logging.error(f"Failed to read analog port {port_number}: {e}")
