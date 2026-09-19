import cv2
import numpy as np
import matplotlib.pyplot as plt
from red_neuronal import RedNeuralAdaptativa
from control_pid import ControlPID

def procesar_fotograma_carta(frame):
    """
    Detecta, extrae y normaliza la carta del fotograma de la cámara.
    Retorna el vector plano de 1024 elementos y el centro (X, Y).
    """
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centro_carta = None
    vector_caracteristicas = None
    
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 1000:  # Filtrar ruido por tamaño mínimo
            perimetro = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimetro, True)
            
            if len(approx) == 4: # Contorno rectangular aproximado de una carta
                x, y, w, h = cv2.boundingRect(approx)
                carta_recorte = frame[y:y+h, x:x+h]
                
                if carta_recorte.size > 0:
                    carta_redimensionada = cv2.resize(carta_recorte, (32, 32))
                    carta_gris_roi = cv2.cvtColor(carta_redimensionada, cv2.COLOR_BGR2GRAY)
                    
                    # Normalizar a rango [0, 1] y aplanar (1, 1024)
                    vector_caracteristicas = carta_gris_roi.flatten().astype(np.float32) / 255.0
                    vector_caracteristicas = vector_caracteristicas.reshape(1, -1)
                    
                    # Calcular centro geométrico (cX, cY) para el control
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        centro_carta = (cX, cY)
                break 
                
    return vector_caracteristicas, centro_carta, thresh

if __name__ == "__main__":
    np.random.seed(42)
    
    # 1. Instanciar módulos independientes
    nn = RedNeuralAdaptativa(input_size=1024, hidden_size=64, output_size=4, learning_rate=0.01)
    
    Kp = np.array([1.2, 1.2, 1.0])
    Ki = np.array([0.05, 0.05, 0.01])
    Kd = np.array([0.15, 0.15, 0.10])
    pid = ControlPID(Kp, Ki, Kd)

    # Abrir la cámara
    cap = cv2.VideoCapture(0)
    
    print("Sistema de Visión + PID + Red Neuronal iniciado.")
    print(" - Presiona 't' para entrenar con la carta actual y registrar pérdida.")
    print(" - Presiona 'q' para salir y generar la gráfica de convergencia.")

    X_dataset = []
    y_dataset = []
    loss_history = []  # Lista para almacenar la evolución de la pérdida

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error al acceder a la cámara.")
            break
            
        vector, centro, vista_binaria = procesar_fotograma_carta(frame)
        
        if vector is not None and centro is not None:
            cv2.circle(frame, centro, 5, (0, 255, 0), -1)
            
            # Inferencia de la carta detectada
            clase, confianza = nn.predecir_carta(vector)
            texto_estado = f"Carta: Clase {clase} ({confianza*100:.1f}%)"
            cv2.putText(frame, texto_estado, (centro[0] - 40, centro[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Error simulado para el PID basado en la posición de la carta
            error_actual = np.array([[centro[0] - 320, centro[1] - 240, 0.0]]) / 100.0
            u_pid = pid.calcular(error_actual)
            u_neural = nn.forward(vector)
            # u_total = u_pid + u_neural  # Señal combinada para los motores

        # Mostrar ventanas de trabajo
        cv2.imshow("Camara - Brazo Delta", frame)
        cv2.imshow("Procesamiento Binario", vista_binaria)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Entrenamiento interactivo al presionar 't'
        if key == ord('t') and vector is not None:
            clase_real = 0  # Ajusta esta clase según la carta que estés mostrando
            target_one_hot = np.zeros((1, 4))
            target_one_hot[0, clase_real] = 1.0
            
            X_dataset.append(vector[0])
            y_dataset.append(target_one_hot[0])
            print(f"Muestra guardada. Total dataset: {len(X_dataset)}")
            
            if len(X_dataset) >= 5:
                X_train = np.array(X_dataset)
                y_train = np.array(y_dataset)
                
                # Entrenar por épocas y registrar cada pérdida
                for epoch_idx in range(50):
                    loss = nn.train_step(X_train, y_train)
                    loss_history.append(loss)  # Guardar historial para la gráfica
                    
                print(f"Entrenamiento completado | Última Pérdida (Loss): {loss:.6f}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- GENERAR GRÁFICA DE ADAPTACIÓN AL SALIR ---
    if len(loss_history) > 0:
        print("Generando gráfica de convergencia de la red...")
        plt.figure(figsize=(9, 5))
        plt.plot(loss_history, color='blue', linewidth=2, label='Cross-Entropy Loss')
        plt.title('Evolución y Adaptación de la Red Neuronal de Visión')
        plt.xlabel('Pasos de Entrenamiento / Épocas acumuladas')
        plt.ylabel('Pérdida (Loss)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()
    else:
        print("No se registraron datos de entrenamiento para graficar.")