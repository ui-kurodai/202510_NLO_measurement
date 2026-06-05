import logging
import serial
from optosigma.gsc02 import GSC02
import optosigma.gsc02 as gsc_module

from typing import Tuple, Union, Sequence, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - in %(filename)s - %(message)s')


class GSC02Controller(GSC02):
    def __init__(self, *args, **kwargs):
        # self._is_connected = False
        self._is_energized = True
        super().__init__(*args, **kwargs)

    # sometimes this is not called depending on MRO (Method Resolution Order)
    # in that case, GSC02.open() is called.
    # def open(self):
    #     try:
    #         super().open()
    #         self._is_connected = True
    #         # print(f"self class: {self.__class__}")
    #         # print(f"[OPEN] id={id(self)} connected port: {gsc_module._gsc02_opened_objects}\n")
    #     except (serial.SerialException, OSError) as e:
    #         self._is_connected = False
    #         logging.error(f"Failed to open serial port: {e}")

    def close(self):
        try:
            super().close()
            # self._is_connected = False
            logging.info('Stage connection safely closed')
        except (serial.SerialException, OSError) as e:
            logging.error(f"Failed to close serial port: {e}")

    @property
    def is_connected(self) -> bool:
        # return self._is_connected
        return self.port in gsc_module._gsc02_opened_objects
    
    @property
    def is_energized(self) -> bool:
        return self._is_energized

    def raw_command(self, command: str) -> Union[bool, Any]:
        try:
            return super().raw_command(command)
        except serial.SerialException:
            logging.error("Serial communication error during raw_command.")
        except Exception:
            logging.error(f"Failed to execute raw_command: {command}")

    def _get_position(self, axis: Union[int, str]) -> int:
        try:
            return super()._get_position(axis)
        except Exception:
            logging.error(f"Failed to get position for axis {axis}.")

    def _set_position(self, target_position: int, axis: Union[int, str]) -> bool:
        try:
            super()._set_position(target_position, axis)
            self.sleep_until_stop()
        except Exception:
            logging.error(f"Failed to set position for axis {axis}.")

    @property
    def position1(self) -> int:
        try:
            return super().position1
        except Exception:
            logging.error("Failed to get position1.")

    @position1.setter
    def position1(self, target_position):
        try:
            super().position1(target_position)
        except Exception:
            logging.error("Failed to set position1.")

    @property
    def position2(self) -> int:
        try:
            return super().position2
        except Exception:
            logging.error("Failed to get position2.")

    @position2.setter
    def position2(self, target_position):
        try:
            super().position2(target_position)
        except Exception:
            logging.error("Failed to set position2.")

    @property
    def ack1(self) -> str:
        try:
            return super().ack1
        except AttributeError as e:
            logging.error(f'Failed to excecute ack1:{e}')
            return None

    @property
    def ack2(self) -> str:
        try:
            return super().ack2
        except AttributeError as e:
            logging.error(f'Failed to excecute ack2:{e}')
            return None

    @property
    def ack3(self) -> str:
        try:
            return super().ack3
        except AttributeError as e:
            logging.error(f'Failed to excecute ack3:{e}')
            return None

    @property
    def is_ready(self) -> bool:
        try:
            return super().is_ready
        except Exception as e:
            logging.error(f'Failed to check if stage is ready: {e}')
            return False

    @property
    def is_last_command_success(self) -> bool:
        try:
            ret = super().is_last_command_success
            return ret
        except Exception as e:
            logging.error(f'Failed to check if the last command is accepted')
            return False
        
    def sleep_until_stop(self):
        try:
            super().sleep_until_stop()
        except Exception as e:
            logging.error(f'Failed to sleep until stop:{e}')


    def return_origin(self, direction: Union[str, Sequence[str]], axis: Union[int, str]) -> bool:
        try:
            ret = super().return_origin(direction=direction, axis=axis)
            self.sleep_until_stop()
            return ret
        except Exception:
            logging.error(f"Failed to return to origin (axis={axis}, direction={direction}).")
        
    def set_relative_pulse(self, pulse: Union[int, Sequence[int]], axis: Union[int, str]) -> bool:
        try:
            ret = super().set_relative_pulse(pulse, axis=axis)
            self.sleep_until_stop()
            return ret
        except Exception:
            logging.error("Failed to set relative pulse.")

    def jog(self, direction: Union[str, Sequence[str]], axis: Union[int, str]) -> bool:
        try:
            return super().jog(direction=direction, axis=axis)
        except Exception:
            logging.error("Failed to jog.")

    def driving(self) -> bool:
        try:
            return super().driving()
        except Exception:
            logging.error("Failed to drive.")

    def decelerate_stop(self, axis: Union[int, str]) -> bool:
        try:
            ret = super().decelerate_stop(axis=axis)
            self.sleep_until_stop()
            return ret
        except Exception:
            logging.error("Failed to decelerate stop.")

    def immediate_stop(self) -> bool:
        try:
            ret = super().immediate_stop()
            self.sleep_until_stop()
            return ret
        except Exception:
            logging.error("Failed to immediate stop.")

    def set_logical_zero(self, axis: Union[int, str]) -> bool:
        try:
            ret = super().set_logical_zero(axis=axis)
            self.sleep_until_stop()
            return ret
        except Exception:
            logging.error("Failed to set logical zero.")

    def set_speed(self, axis: Union[int, str], spd_min: int, spd_max: int, acceleration_time: int) -> bool:
        try:
            axis_key = str(axis).upper()
            spd_min = int(spd_min)
            spd_max = int(spd_max)
            acceleration_time = int(acceleration_time)
            if axis_key not in {"1", "2", "W"}:
                raise ValueError("Speed axis must be 1, 2, or W.")
            if not 1 <= spd_min <= 30000:
                raise ValueError("Minimum speed must be between 1 and 30000 pps.")
            if not 1 <= spd_max <= 30000:
                raise ValueError("Maximum speed must be between 1 and 30000 pps.")
            if not 0 <= acceleration_time <= 1000:
                raise ValueError("Acceleration time must be between 0 and 1000 ms.")
            if spd_min > spd_max:
                raise ValueError("Minimum speed must be less than or equal to maximum speed.")

            params = f"S{spd_min}F{spd_max}R{acceleration_time}"
            command = f"D:{axis_key}{params}{params if axis_key == 'W' else ''}"
            ret = self.raw_command(command)
            self.sleep_until_stop()
            return ret
        except Exception as e:
            logging.error(f"Failed to set speed: {e}")

    def energize_motor(self, energize: bool, axis: int) -> bool:
        try:
            ret = super().energize_motor(energize, axis=axis)
            self.sleep_until_stop()
            self._is_energized = energize
            return ret
        except Exception:
            logging.error("Failed to energize motor.")

    def get_status1(self) -> Tuple[int, int, str, str, str]:
        try:
            return super().get_status1()
        except Exception:
            logging.error("Failed to get status1.")

    def get_status2(self) -> str:
        try:
            return super().get_status2()
        except Exception:
            logging.error("Failed to get operating status.")

    def get_version(self) -> str:
        try:
            return super().get_version()
        except Exception:
            logging.error("Failed to get version.")
