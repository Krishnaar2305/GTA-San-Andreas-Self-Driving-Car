class PID:
    def __init__(self, Kp, Ki, Kd, mn=-16384, mx=16384, neutral=16384):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.mn = mn
        self.mx = mx
        self.integral = 0.0
        self.prev_error = 0.0
        self.neutral = neutral

    def control(self, error, dt):
        if dt <= 0:
            return 0.0

        # Compute integral and derivative
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        # Raw PID output
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Anti-windup + clamp
        clamped_output = max(self.mn, min(self.mx, output))
        if clamped_output != output:
            self.integral -= error * dt
        output = clamped_output

        # Store for next iteration
        self.prev_error = error
        return int(output + self.neutral)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        return