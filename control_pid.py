import numpy as np

class ControlPID:
    def __init__(self, kp, ki, kd):
        self.Kp = np.array(kp, dtype=float)
        self.Ki = np.array(ki, dtype=float)
        self.Kd = np.array(kd, dtype=float)
        
        self.integral = 0.0
        self.error_anterior = 0.0

    def calcular(self, error_actual):
        # Acumulador integral y cálculo de la derivada
        self.integral += error_actual
        derivada = error_actual - self.error_anterior
        self.error_anterior = error_actual
        
        # Ley de control PID: Proporcional + Integral + Derivativa
        u = (self.Kp * error_actual) + (self.Ki * self.integral) + (self.Kd * derivada)
        return u