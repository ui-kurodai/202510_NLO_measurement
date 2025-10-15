import logging
import serial
from typing import Optional
from devices.gsc02_control import GSC02Controller

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - in %(filename)s - %(message)s')


class OSMS2035Controller(GSC02Controller):
    is_sleep_until_stop = True
    mm_per_pulse = 0.001  # [mm/pulse] (fixed)

    def __init__(
        self,
        port=None,
        baudrate=9600,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=None,
        xonxoff=False,
        rtscts=False,
        write_timeout=None,
        dsrdtr=False,
        inter_byte_timeout=None,
        exclusive=None,
        axis=1,
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
        logging.info(f"OSMS2035Controller initialized on port {port} (axis {axis})")


    def reset(self, direction="-") -> bool:
        try:
            self.return_origin(direction, axis=self.axis)
        
        except (serial.SerialTimeoutException, serial.SerialException, UnicodeDecodeError) as e: # catch SerialTimeoutException before SerialException 
            logging.error(f'Failed to reset stage{self.axis}')


    def stop(self) -> bool:
        return self.decelerate_stop(axis=self.axis)
  

    @property
    def millimeter(self) -> Optional[float]:
        try:
            pulses = self._get_position(self.axis)
            return self.pos2mm(pulses)
        except Exception as e:
            logging.error(f"Failed to get position in mm: {e}")


    @millimeter.setter
    def millimeter(self, mm: float):
        try:
            pulses = self.mm2pos(mm)
            self._set_position(pulses, self.axis)
            self.sleep_until_stop()

            delta = mm - self.millimeter
            if abs(delta) > 0.002:
                logging.warning(f"Position error at {mm} mm by {delta} mm")
                
        except Exception as e:
            logging.error(f"Failed to set position in mm: {e}")


    def pos2mm(self, pos: int) -> float:
        try:
            return pos * self.mm_per_pulse
        except Exception as e:
            logging.error(f"pos2mm conversion failed: {e}")


    def mm2pos(self, mm: float) -> int:
        try:
            return int(mm / self.mm_per_pulse)
        except Exception as e:
            logging.error(f"mm2pos conversion failed: {e}")
