import math


class CameraModel:
    def __init__(
        self,
        width: int,
        height: int,
        fx: float = 512.056,
        fy: float = 456.726,
        yaw_per_count: float = 0.00219291,
        pitch_per_count: float = 0.00244552,
    ):
        self.width = width
        self.height = height
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.fx = fx
        self.fy = fy
        self.yaw_per_count = yaw_per_count
        self.pitch_per_count = pitch_per_count

    def screen_to_mouse(self, x: float, y: float):
        yaw = math.atan((x - self.cx) / self.fx)
        dx = round(yaw / self.yaw_per_count)
        # dy 为正 = 视角向下，对应屏幕坐标向下
        pitch = math.atan((y - self.cy) / self.fy)
        dy = round(pitch / self.pitch_per_count)
        return dx, dy

    def screen_error(self, x: float, y: float):
        return (x - self.cx, y - self.cy)

    def project_independent(self, x, y, mouse_dx, mouse_dy):
        """Original independent-axis projection, also used for offline replay."""
        return (
            self.cx
            + self.fx
            * math.tan(math.atan((x - self.cx) / self.fx) - mouse_dx * self.yaw_per_count),
            self.cy
            + self.fy
            * math.tan(math.atan((y - self.cy) / self.fy) - mouse_dy * self.pitch_per_count),
        )
