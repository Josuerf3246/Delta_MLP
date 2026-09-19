import numpy as np

class ControlPID:
    def __init__(self, Kp, Ki, Kd):

        # variables con el prefijo self. que nos ayuda a guardarlas en la memoria
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        # Aqui se almacena la memoria de los errores acumulados dentro del termino integral y derivativo.
        self.integral_error = np.zeros((1, 3))
        self.previous_error = np.zeros((1, 3))

    def calcular(self, error_actual):
        self.integral_error += error_actual
        derivative_error = error_actual - self.previous_error

    # Este es el control, como sabemos kp*error actual + ki*error integral + kd*error derivativo, una accion de control que se ejecuta mediante la actualizacion de las ganancias.
    # Puesto que la red neuronal es de aprendizaje y actualiza las ganancias para hacer mejores movimientos cada iteracion.
        u_pid = (self.Kp * error_actual) + (self.Ki * self.integral_error) + (self.Kd * derivative_error)


        self.previous_error = error_actual
        return u_pid