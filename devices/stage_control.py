from optosigma import OSMS60YAW, OSMS2085
import time

DEFAULT_PORT = "COM4"

class StageController:
    def __init__(self, port=DEFAULT_PORT, rot_axis=2, lin_axis=1):
        self.port = port
        self.rot_axis = rot_axis
        self.lin_axis = lin_axis
        self.rotation_stage = OSMS60YAW(port=self.port, axis=self.rot_axis)
        self.translation_stage = OSMS2085(port=self.port, axis=self.lin_axis)
        self.rotation_stage.open()
        self.translation_stage.open()

    def _safe(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[WARNING] Stage command failed: {e}")
            try:
                self.rotation_stage.close()
                self.translation_stage.close()
            except:
                pass
            try:
                self.rotation_stage.open()
                self.translation_stage.open()
                print("[INFO] Reconnection successful. Retrying...")
                return func(*args, **kwargs)
            except Exception as e2:
                print(f"[ERROR] Retry failed: {e2}")
                return None

    # --- Rotation stage methods ---
    def move_to_angle(self, deg: float):
        return self._safe(setattr, self.rotation_stage, "degree", deg)

    def get_angle(self) -> float:
        return self._safe(getattr, self.rotation_stage, "degree")

    def reset_rotation(self):
        return self._safe(self.rotation_stage.reset)

    # --- Translation stage methods ---
    def move_to_mm(self, mm: float):
        return self._safe(setattr, self.translation_stage, "millimeter", mm)

    def get_position_mm(self) -> float:
        return self._safe(getattr, self.translation_stage, "millimeter")

    def reset_translation(self):
        return self._safe(self.translation_stage.reset)

    def close_all(self):
        self.rotation_stage.close()
        self.translation_stage.close()

    def __del__(self):
        self.close_all()