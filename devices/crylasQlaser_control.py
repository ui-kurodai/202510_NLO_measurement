import serial
import time
import re
import logging
from typing import Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - in %(filename)s - %(message)s')

class CrylasQLaserDecoder:

# Error bitmask definitions (from manual Table 8)
    ERROR_BITS = {
        0:  "Supply voltage too low",
        1:  "Supply voltage too high",
        2:  "TEC1 temperature out of range",
        3:  "TEC2 temperature out of range",
        4:  "TEC1 temperature not adjustable",
        5:  "TEC2 temperature not adjustable",
        6:  "Receive Time-Out",
        7:  "Unknown command",
        8:  "Command forbidden in this mode",
        9:  "Parameter out of range",
        10: "Too many symbols in command",
        11: "No connection to laser head",
        12: "LD power supply too low",
        13: "External freq too high → laser off",
        17: "TEC1 voltage overload",
        18: "TEC2 voltage overload",
        19: "Wrong controller type",
        20: "LD voltage out of range",
        21: "TEC1 temp again out of range",
        22: "TEC2 temp again out of range",
        23: "TEC1 current overload",
        24: "TEC2 current overload",
        25: "No comm. with A/D converter",
        26: "No comm. with laser processor",
        27: "No comm. with TEC processor",
        28: "No comm. with modulation processor",
        29: "No comm. with controller memory",
        30: "Laser head temp out of range",
        31: "No comm. with laser head memory",
        32: "Laser head memory out of range"
    }

    def status_byte_decoded(self, list_bytes) -> dict:
        """
        Decode the response from status() method
        the return of status() is:
        Format: ['>vs', '<operation-byte>', '<status-byte>']
        Example: ['>vs', '0D', '3C']
        """
        try:
            op_hex, st_hex = list_bytes[1:]
            op = int(op_hex, 16)
            st = int(st_hex, 16)

            # mapping the bytes on messages (Table 6)
            mapping = {
                0: ("TEC1_On",       "TEC1_OK"),
                1: ("TEC2_On",       "TEC2_OK"),
                2: ("Laser_On",      "Laser_OK"),
                3: ("Modulation_On", "Modulation_On"),
                4: ("Master_Mode",   "Laser_ready"),
                5: ("Debug_Mode",    "Set_parameters"),
                6: ("Mem_Access",    "Interlock_open"),
                7: (None,            "Fatal_error"),
            }

            flags = {}
            for bit, (op_name, st_name) in mapping.items():
                mask = 1 << bit
                if op_name:
                    flags[op_name] = bool(op & mask)
                if st_name:
                    flags[st_name] = bool(st & mask)

            return {
                "operation_byte": op,
                "status_byte":    st,
                "flags":          flags,
            }
        except (ValueError, TypeError) as e:
            logging.error(f'Unexpected laser status returned:{list_bytes}, {e}')
            return None

    # ===== Error decoding =====
    def get_error_decoded(self, error_list) -> dict:
        """
        Decode the response from 'last_error_bytes' method
        the return of the method is:
        Format: ['>er', '<decimal-byte>', '<hex-byte>']
        Example: ['>er', '0', '00000000']
        """
        if not error_list:
            return {"error": "No response"}
        try:
            error_byte = error_list[2]
            if not error_byte:
                return {"error": f"Hex bitmask not found in response: {error_list}"}

            bitmask = int(error_byte, 16)

            decoded = {
                desc: bool(bitmask & (1 << bit))
                for bit, desc in self.ERROR_BITS.items()
            }
            return decoded
        except Exception as e:
            return {"error": f"Parse failed: {e}"}

class CrylasQLaserController:

    def __init__(self):
        self._ser = None
        self._delimiter = "\r"
        self._delimiter_bytes = self._delimiter.encode()
             
    def open(self, port, baudrate=19200, timeout=1.0):
        try:
            self._ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                stopbits=serial.STOPBITS_ONE,
                parity=serial.PARITY_NONE,
                timeout=timeout
            )
            self._info_control_unit()
            self._info_laser_head()
        except (serial.SerialException, OSError) as e:
            logging.error(f"Failed to open serial port {port}: {e}")

    def close(self):
        try:
            self._ser.close()
            logging.info(f"Laser connection closed")
        except (serial.SerialException, OSError) as e:
            logging.error(f"Failed to close serial port: {e}")

    @property
    def is_connected(self) -> bool:
        return self._ser is not None

    
    def send_command(self, cmd:str):
        command = cmd + self._delimiter
        try:
            # clearing response buffer
            if self._ser.in_waiting > 0:
                self._ser.reset_input_buffer()
            self._ser.write(command.encode())
        except (serial.SerialException, OSError, UnicodeEncodeError) as e:
            logging.error(f"Failed to send command: {e}")
    
    
    def receive_response(self) -> Optional[str]:
        try:
            return self._ser.read_until(self._delimiter_bytes).decode().strip()
        except (serial.SerialTimeoutException, serial.SerialException, UnicodeDecodeError) as e: # catch SerialTimeoutException before SerialException 
            logging.error(f"Failed to receive response: {e}")
            return None

    # ===== Table 5 Commands =====

    def _info_control_unit(self):
        """
        Returns the control unit's ID and firmware version string without the leading 'id' keyword.
        Example:
            "id DX2810-208-2 ..." → "DX2810-208-2 ..."
        """
        try:
            self.send_command("id")
            response = self.receive_response()
        except (serial.SerialTimeoutException, serial.SerialException, OSError) as e:
            logging.error(f"Failed to read control unit info: {e}")
            return
        self._firmware = response.replace("id", "").replace(" ", "")

    

    def _info_laser_head(self):
        """
        Parse laser head info from 'im' command.
        Format: '>im <type> <serial> <date> <optional_info>'
        Example: '>im 106403__ 00002233 160424 ________________'
        
        laser type (8 characters), serial number (8 characters), shipping date (6 characters)
        when the command returns error.
        '>im ERROR 8 00000000'
        """
        try:
            self.send_command("im")
            response = self.receive_response()
        except (serial.SerialTimeoutException, serial.SerialException, OSError) as e:
            logging.error(f"Failed to read laser head info: {e}")
            return
        
        self._laser_info_all = response # debugging
        parts = response.strip().split()
        if len(parts) != 5:
            if "ERROR" in response:
                logging.error(f'"im" command returned error status:{response!r}')
            else:
                logging.error(f"Unexpected 'im' response format: {response!r}")
            self._laser_type = None
            self._serial_number = None
            self._shipping_date = None
            self._additional_info = None
        else:
            self._laser_type = parts[1]
            self._serial_number = parts[2]
            self._shipping_date = parts[3]
            self._additional_info = parts[4]
    
    # debugging with raw output
    @property
    def laser_info(self):
        return self._laser_info_all
    
    @property
    def wavelength_nm(self):
        return 1064.0


    @property
    def firmware(self):
        return self._firmware
    

    @property
    def laser_type(self):
        return self._laser_type
    

    @property
    def serial_number(self):
        return self._serial_number
    

    @property
    def shipping_date(self):
        return self._shipping_date
    

    @property
    def additional_info(self):
        return self._additional_info
    

    @property
    def status(self) -> list[str]:  # this method is doing too much. Better splitting tasks into different methods.
        """
        returns the response to 'vs' command.
        Format: 'vs <operation-byte> <status-byte> \r'
        Example: 'vs 0D 3C\r\n>'
        """
        try:
            self.send_command("vs")   # ex: "vs\r\n>"
            list_byte_str = self.receive_response().split(" ")
        except (serial.SerialTimeoutException, serial.SerialException, OSError) as e:
            logging.error(f"Failed to get status byte: {e}")
            return None
        if len(list_byte_str) == 3:
            return list_byte_str
        else:
            logging.warning(f"Response of wrong format for operation/status bytes")
            return list_byte_str # debugging


    def start(self):
        """
        laser emission starts after 0.4 sec of delay
        """
        try:
            self.send_command("st")
            logging.info("laser on")
        except (serial.SerialException, OSError) as e:
            logging.error(f"Failed to start laser: {e}")


    def stop(self):
        try:
            self.send_command("rs")
            logging.info("laser off")
        except (serial.SerialException, OSError) as e:
            logging.error(f"Failed to stop laser: {e}")


    @property
    def last_error_bytes(self) -> Optional[str]:
        try:
            self.send_command("er")
            decimal_error_code = self.receive_response().split(" ")
            return decimal_error_code
        except (serial.SerialException, OSError, TypeError, IndexError) as e:
            logging.error(f"Failed to display last error code: {e}")
            return None


    def clear_error(self):
        try:
            self.send_command("re")
        except (serial.SerialException, OSError) as e:
            logging.error(f"Failed to clean actual error condition: {e}")


    @property
    def rep_rate(self) -> Optional[int]:
        """
        Parse rep rate from 'mf' command.
        Format: '>mf <set_rep> <unkown1> <unkown2> <unkown3>'
        Example: '>mf 1000 1006 0 0.000'

        The latter two output is not defined.
        unknown1 is probably the actual rep (not precise though). 
        """
        try:
            self.send_command("mf")
            response = self.receive_response()
            response = response.strip().split()
            if response[1].isdigit():
                return int(response[1])
            else:
                return np.nan
        except (serial.SerialTimeoutException, serial.SerialException, OSError, TypeError) as e:
            logging.error(f"Failed to read actual repetition rate for internal trigger mode: {e}")
            return None


    @property
    def max_allowed_rep_rate(self) -> Optional[int]:
        """
        Parse rep rate from 'mm' command.
        Format: '>mm <max_rep> <unkown1>'
        Example:'>mm 2500 1'
        """
        try:
            self.send_command("mm")
            response = self.receive_response()
            response = response.strip().split()
            if response[1].isdigit():
                return int(response[1])
            else:
                return np.nan
        except (serial.SerialTimeoutException, serial.SerialException, OSError, TypeError) as e:
            logging.error(f"Failed to get maximum allowed repetition rate: {e}")
            return None


    @rep_rate.setter
    def rep_rate(self, new_rep_rate: int):
        max_val = self.max_allowed_rep_rate
        if max_val is None:
            return
        if 1 < new_rep_rate <= max_val:
            try:
                self.send_command(f"mf {new_rep_rate}")
            except (serial.SerialException, OSError) as e:
                logging.error(f"Failed to set repeition rate to {new_rep_rate}: {e}")
        else:
            logging.warning(f"Requested rep rate out of range (max rep rate = {max_val})")

    
    @property
    def trigger_mode(self) -> Optional[int]:
        """
        0: Trigger mode is set to "Internal" (falling edge of TTL signal applied to pin 2);
        1: Trigger mode is set to "External" (falling edge of TTL signal applied to pin 2);
        2: Trigger mode is set to "Internal" (controlled by the level of TTL signal applied to pin 2)
        3: Trigger mode is set to "External" (controlled by the level of TTL signal applied to pin 2)
        """
        try:
            self.send_command("et")
            response = self.receive_response()
            response = response.strip().split()
            if response[1].isdigit():
                return int(response[1])
            else:
                return np.nan
        except (serial.SerialTimeoutException, serial.SerialException, OSError, TypeError) as e:
            logging.error(f"Failed to read actual trigger mode: {e}")
            return None


    @trigger_mode.setter
    def trigger_mode(self, mode: int):
        """
        0: Trigger mode is set to "Internal" (falling edge of TTL signal applied to pin 2);
        1: Trigger mode is set to "External" (falling edge of TTL signal applied to pin 2);
        2: Trigger mode is set to "Internal" (controlled by the level of TTL signal applied to pin 2)
        3: Trigger mode is set to "External" (controlled by the level of TTL signal applied to pin 2)
        """
        if mode not in [0, 1, 2, 3]:
            logging.warning(f"Parameter for command 'et' not valide (0, 1, 2, 3): parameter={mode}")
            return
        try:
            self.send_command(f"et {mode}")
        except (serial.SerialException, OSError) as e:
            logging.error(f"Failed to set trigger mode: {e}")
    

    @property
    def operating_hours(self) -> Optional[float]:
        try:
            self.send_command("vt")
            response = self.receive_response()
            response = response.strip().split()
            if response[1].isdigit():
                return int(response[1]) * 0.1
            else:
                return np.nan
        except (serial.SerialTimeoutException, serial.SerialException, OSError, TypeError) as e:
            logging.error(f"Failed to read operating hours: {e}")
            return None
    

    @property
    def number_emitted_pulses(self) -> Optional[int]:
        try:
            self.send_command("ec")
            response = self.receive_response()
            response = response.strip().split()
            if response[1].isdigit():
                return int(response[1])
            else:
                return np.nan
        except (serial.SerialTimeoutException, serial.SerialException, OSError, TypeError) as e:
            logging.error(f"Failed to read number of emitted pulses: {e}")
            return None


    @property
    def photodiode_voltage(self) -> Optional[float]:
        """
        photodiode voltage [mV] from 'di 0008' → 'ga' sequence.
        Returns: voltage as float or None
        """
        try:
            self.send_command("di 0008") # check if this command is refused while streaming
            self.send_command("ga")
            response = self.receive_response()
            response = response.strip().split()
            if response[1].isdigit():
                return float(response[1])
            else:
                return np.nan
        except (serial.SerialTimeoutException, serial.SerialException, OSError, TypeError) as e:
            logging.error(f"Failed to read photodiode voltage: {e}")
            return None


    def start_photodiode_voltage_stream(self): # streaming needs to be tested (maybe buffer overflow happens)
        try:
            self.send_command("di 8008")
        except (serial.SerialException, OSError) as e:
            logging.error(f"Failed to start stream photodiode voltage: {e}")
        # return self.safe_command("di 8008")


    def stop_stream(self):
        # return self.safe_command("di 0000")
        try:
            self.send_command("di 0000")
        except (serial.SerialException, OSError) as e:
            logging.error(f"Failed to stop stream laser output data: {e}")


    # def read_pd_voltage_stream(self, callback: callable):
    #     try:
    #         while self._streaming_active:
    #             response = self.receive_response()
    #             parts = response.strip().split()
    #             for part in parts:
    #                 value = part
    #                 callback(value)
    #     except (serial.SerialException, OSError, UnicodeDecodeError) as e:
    #         logging.error(f"Error during PD stream read: {e}")
    #     finally:
    #         self.stop_stream()
    #         logging.info("Stream safely stopped.")


    # consider if these two methods will be implemented or not...
    # def request_all_params(self):
    #     self.safe_command("di 7FFF")
    #     return self.safe_command("ga", wait=0.5)


    # def get_all_parameters(self, decoded=False):
    #     """
    #     Get all diagnostic parameters from 'di 7FFF' → 'ga' and decode them into readable form.
    #     """
    #     self.safe_command("di 7FFF")
    #     raw = self.safe_command("ga", wait=0.5)
    #     if not decoded or not raw:
    #         return raw
    #     try:
    #         # extract all signed/unsigned integers from raw response
    #         values = [int(x) for x in re.findall(r"-?\d{1,6}", raw)]
    #         return {
    #             "case_temp_C":        values[0] / 1000,
    #             "set_TEC1_temp_C":    values[1] / 1000,
    #             "real_TEC1_temp_C":   values[2] / 1000,
    #             "set_TEC2_temp_C":    values[3] / 1000,
    #             "real_TEC2_temp_C":   values[4] / 1000,
    #             "TEC1_percent":       values[5],
    #             "TEC2_percent":       values[6],
    #             "PD_voltage_mV":      values[7],
    #             "PD_target_mV":       values[8],
    #             "LD_target_mA":       values[9],
    #             "LD_actual_mA":       values[10],
    #             "LD_operation_sec":   values[11],
    #             "operation_byte":     values[12],
    #             "status_byte":        values[13],
    #             "supply_voltage_mV":  values[14],
    #             "interlock_state":    values[15],
    #             "limit_flag":         values[16],
    #             "power_reached":      values[17]
    #         }
    #     except Exception as e:
    #         return {"error": f"Failed to parse: {e}"}

    @property
    def is_emission_on(self) -> bool:
        try:
            decoder = CrylasQLaserDecoder()
            decoded = decoder.status_byte_decoded(self.status)
            flags = decoded.get("flags", {})
            return flags.get("Laser_On", False)
        except Exception as e:
            logging.warning(f"Failed to decode laser emission status: {e}")
            return False
        
    @property
    def is_fatal_error(self) -> bool:
        try:
            decoder = CrylasQLaserDecoder()
            decoded = decoder.status_byte_decoded(self.status)
            flags = decoded.get("flags", {})
            return flags.get("Fatal_error", False)
        except Exception as e:
            logging.warning(f"Failed to decode laser emission status: {e}")
            return False
        
    @property
    def last_error_message(self) -> dict:
        try:
            error_bytes = self.last_error_bytes
            decoder = CrylasQLaserDecoder()
            decoded = decoder.get_error_decoded(error_bytes)
            error = []
            error.extend([key for key, value in decoded.items() if value])
            return error
        except Exception as e:
            logging.warning(f"Failed to decode error bytes: {e}")
            return {"error": f"Failed to decode error bytes: {error_bytes}"}

    def __del__(self):
        self.close()
        

