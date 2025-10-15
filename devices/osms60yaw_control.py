import logging
import serial
from typing import Optional
from devices.gsc02_control import GSC02Controller

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - in %(filename)s - %(message)s')


class OSMS60YAWController(GSC02Controller):
    is_sleep_until_stop = True
    degree_per_pulse = 0.0025  # [deg/pulse] (fixed)

    def __init__(
        self,
        port=None,
        baudrate=9600,
        bytesize=8,
        parity='N',
        stopbits=1,
        timeout=None,
        xonxoff=False,
        rtscts=False,
        write_timeout=None,
        dsrdtr=False,
        inter_byte_timeout=None,
        exclusive=None,
        axis=2,
        **kwargs,
    ):
        super().__init__(
            port=port,
            baudrate=baudrate,
            bytesize=bytesize,
            parity=parity,
            stopbits=stopbits,
            timeout=timeout,
            xonxoff=xonxoff,
            rtscts=rtscts,
            write_timeout=write_timeout,
            dsrdtr=dsrdtr,
            inter_byte_timeout=inter_byte_timeout,
            exclusive=exclusive,
            **kwargs,
        )
        self.axis = axis  # 1 or 2
        logging.info(f"OSMS60YAWController initialized on port {port} (axis {axis})")


    def reset(self, direction='-') -> bool:
        try:
            self.return_origin(direction, axis=self.axis)
            
        except (serial.SerialTimeoutException, serial.SerialException, UnicodeDecodeError) as e: # catch SerialTimeoutException before SerialException 
            logging.error(f'Failed to reset stage{self.axis}')


    def stop(self) -> bool:
        return self.decelerate_stop(axis=self.axis)


    @property
    def degree(self) -> Optional[float]:
        try:
            pulses = self._get_position(self.axis)
            return self.pos2deg(pulses)
        except Exception as e:
            logging.error(f"Failed to get position in degrees: {e}")


    @degree.setter
    def degree(self, deg: float):
        try:
            pulses = self.deg2pos(deg)
            self._set_position(pulses, self.axis)
            self.sleep_until_stop()
        except Exception as e:
            logging.error(f"Failed to set position in degrees: {e}")


    def pos2deg(self, pos: int) -> float:
        try:
            return pos * self.degree_per_pulse
        except Exception as e:
            logging.error(f"pos2deg conversion failed: {e}")


    def deg2pos(self, deg: float) -> int:
        try:
            return int(deg / self.degree_per_pulse)
        except Exception as e:
            logging.error(f"deg2pos conversion failed: {e}")


    def move_to_angle(self, target_deg: float, direction: str = "auto"):
        try:
            if not 0 <= target_deg <360:
                logging.error("Invalid target angle: 0 <= target < 360")
                return
            
            current_deg = self.degree
            if current_deg < 0:
                current_deg = 360 + current_deg
            elif current_deg >= 360:
                current_deg = current_deg % 360

            if not 0 <= current_deg <360:
                logging.error(f"Unexpected current position returned: {self.degree}")
                return

            delta = target_deg - current_deg

            _ = True
            while _:
                if direction == "auto":
                    if -180 <= delta < 0:
                        direction = "cw"
                    else:
                        direction = "ccw"

                # ccw is "+"
                elif direction == "ccw":
                    angle = delta if delta >= 0 else 360 + delta
                    _ = False

                # cw is "-"
                elif direction == "cw":
                    if delta >= 0 or (357 < current_deg < 360):
                        logging.exception("Cannot rotate cw across -2.5 deg")
                        direction = "ccw"
                    else:
                        angle = delta
                        _ = False

                else:
                    logging.error("Invalid direction: choose from 'cw', 'ccw', or 'auto'")
                    angle = 0
                    _ = False

            pulse = self.deg2pos(angle)
            logging.info(f"relative pulse: {pulse}")
            logging.info(f"set relative pulse: {self.set_relative_pulse(pulse, axis=self.axis)}, axis: {self.axis}")
            logging.info(f"driving: {self.driving()}")
            self.sleep_until_stop()
            logging.info(f"Moved to {target_deg:.2f}° (direction: {direction})")
        except Exception as e:
            logging.error(f"Failed to move to specified angle: {e}.")


    def __del__(self):
        self.close()
        logging.info('osms60yaw is safely closed')