# ProyectoA - Detector de matrículas desde fichero de vídeo o desde WebCam en tiempo real
from os.path import exists
import cv2
import matplotlib.pyplot as plt 
import imutils
import numpy as np
import easyocr
import datetime
import re

# Para acelerar el proceso, podemos analizar algunos de los frames y no todos
# Un ratio muy alto puede hacer perder captura de información
ratioAnalisisFrame = 3

# Carpeta donde se guardarán las imágenes de frames con matrículas detectadas
carpetaCapturas = "D:\\Mis documentos\\ProyectoA\\Python\\matricula\\capturas\\"

# Fichero de vídeo con grabación de coches y matrículas
ficheroVideo = "D:\\Mis documentos\\ProyectoA\\Python\\matricula\\matriculas_tiempo_real\\matricula.mp4"

# Cambiaremos a True esta variable para detectar matrículas desde la WebCam en tiempo real
# Dejaremos a False para detectar matrículas del vídeo pasado en ficheroVideo
capturarWebCam = True

# Método que quitará todos los caracteres salvo las letras y los números
def limpiarMatricula(matricula):
    matriculaLimpia = re.sub(r'[^a-zA-Z0-9]', '', matricula)
    # Convertimos a mayúsculas
    matriculaLimpia = matriculaLimpia.upper()
    return matriculaLimpia

# Según el formato del valor obtenido decidirá si es una matrícula o no
# Serán válidas las expresiones regulares: 1234AAA y AA1234AA
# Dejamos comentados otros formatos de matrículas, por si se quieren usar
def validarMatricula(matricula):
    matriculaCorrecta = False
    patron = r'[0-9]{4}[A-Z]{3}$' #1234ABC Unión Europea
    matriculaCorrecta = re.match(patron, matricula)
    if not matriculaCorrecta:
        patron = r'[A-Z]{2}[0-9]{4}[A-Z]{2}$' #AA1234BB Antigua matrícula de España
        matriculaCorrecta = re.match(patron, matricula)    
    # Por si queremos dar como válidos otros formatos
    """
    if not matriculaCorrecta:
        patron = r'[0-9]{4}[A-Z]{2}$' #1234AA
        matriculaCorrecta = re.match(patron, matricula)
    if not matriculaCorrecta:
        patron = r'[A-Z]{2}[0-9]{4}$' #AA1234
        matriculaCorrecta = re.match(patron, matricula)
    if not matriculaCorrecta:
        patron = r'[A-Z]{3}[0-9]{4}$' #AAA1234
        matriculaCorrecta = re.match(patron, matricula)        
    if not matriculaCorrecta:
        patron = r'[A-Z]{2}[0-9]{2}[A-Z]{3}$' #AA12AAA Inglaterra
        matriculaCorrecta = re.match(patron, matricula)
    if not matriculaCorrecta:
        patron = r'[A-Z]{1}[0-9]{3}[A-Z]{2}[0-9]{2}$' #A123AA12 Rusia
        matriculaCorrecta = re.match(patron, matricula)        
    if not matriculaCorrecta:
        patron = r'[0-9]{3}[A-Z]{3}$' #123AAA Estados Unidos
        matriculaCorrecta = re.match(patron, matricula)        
    if not matriculaCorrecta:
        patron = r'[0-9]{1}[A-Z]{3}[0-9]{3}$' #1AAA123 Estados Unidos (segunda opción)
        matriculaCorrecta = re.match(patron, matricula)
    if not matriculaCorrecta:
        patron = r'[0-9]{2}[A-Z]{2}[0-9]{4}[A-Z]{1}$' #12AA1234A India
        matriculaCorrecta = re.match(patron, matricula)
    if not matriculaCorrecta:
        patron = r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$' #AA12AA1234 India (segunda opción)
        matriculaCorrecta = re.match(patron, matricula)
    """
    
    return matriculaCorrecta

if capturarWebCam:
    # Para capturar vídeo de la webcam del equipo en tiempo real y detectar matrículas
    capturaVideo = cv2.VideoCapture(0)
else:    
    # Si queremos capturar vídeo en fichero del equipo y detectar matrículas
    if exists(ficheroVideo):
        capturaVideo = cv2.VideoCapture(filename=ficheroVideo)
    else:
        print("No se ha encontrado el vídeo indicado.")
        exit()

matriculaAnterior = ""

# Si se inicia correctamente la captura de vídeo/webcam
if capturaVideo.isOpened():
    if not capturarWebCam:
        numeroFrames = int(capturaVideo.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Número de frames del vídeo: {numeroFrames}")    

    frameActual = 0
    frameAnalizado = 0
    while capturaVideo.isOpened():
        camaraIniciada, imagenCapturada = capturaVideo.read()
        if not camaraIniciada:
            print("No se ha encontrado imagen en la entrada de WebCam o se ha finalizado el fichero.")
            exit() # para vídeo desde fichero
            # continue # para vídeo desde WebCam
        
        # Capturamos cada frame del vídeo para detectar una posible matrícula
        frameActual += 1
        # Para depurar, mostrar el frame actual analizado
        # print(f"Analizando frame {frameActual} de {numeroFrames}")
        
        if frameActual > frameAnalizado + ratioAnalisisFrame:
            frameAnalizado = frameActual           
            imagenCapturada = cv2.flip(imagenCapturada, 0)
            imagenCapturada = cv2.cvtColor(cv2.flip(imagenCapturada, 0), cv2.COLOR_BGR2RGB)
            imagen = imagenCapturada
            
            # Transformamos la imagen a escala de grises
            imagenEscalaGrises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

            # Aplicamos filtro bilateral (limpia la imagen de posibles ruidos)
            # Esto hará que el paso de detección de bordes sea más preciso
            # Los parámetros 11, 17, 17 determinan el diámetro de la vecindad de píxeles, sigmaColor y sigmaSpace respectivamente
            imagenLimpia = cv2.bilateralFilter(imagenEscalaGrises, 11, 17, 17)

            # Detectar contornos de la imagen
            # Busca en la foto lugares donde el brillo o el color cambian bruscamente
            imagenContornos = cv2.Canny(imagenLimpia, 30, 200)

            # Detectar curvas de nivel
            # Son los límites de los componentes conectados en la imagen
            imagenCurvasNivel = cv2.findContours(imagenContornos.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Ordenamos los contornos en función de su área
            # Se seleccionan los 5 contornos más grandes
            contornosMasGrandes = imutils.grab_contours(imagenCurvasNivel)
            contornosMasGrandes = sorted(contornosMasGrandes, key=cv2.contourArea, reverse=True)[:5]
            
            # Esto no es necesario, solo se muestran los contornos obtenidos para depurar
            # Muestra los contornos en el vídeo en tiempo real
            imagenContornosDibujados = cv2.drawContours(imagen, contornosMasGrandes, -1, (0, 255, 0), 3)

            # La matrícula tendrá una forma rectangular, por lo que recorreremos los contornos obtenidos
            # para seleccionar el contorno que se parezca a un rectángulo (4 puntos)
            rectanguloLocalizado = []
            for contorno in contornosMasGrandes:        
                # Para usar un epsilon fijo: 10
                contornoAproximado = cv2.approxPolyDP(contorno, 10, True)
                
                # Si el contorno aproximado tiene 4 "vértices" suponemos que es la matrícula
                if len(contornoAproximado) == 4:
                    rectanguloLocalizado = contornoAproximado               
                    break

            # Si se ha encontrado una forma rectangular de cuatro puntos
            if len(rectanguloLocalizado) > 0:
                # Se crea una máscara de la misma forma que la imagen en escala de grises, rellena con ceros
                # El contorno de la matrícula encontrada se dibuja en esta máscara
                # Con la máscara, la matrícula real se extrae de la imagen original
                mascara = np.zeros(imagenEscalaGrises.shape, np.uint8)
                contornoMatricula = cv2.drawContours(mascara, [rectanguloLocalizado], 0, 255, -1)
                contornoMatricula = cv2.bitwise_and(imagen, imagen, mask=mascara)

                # Con la máscara, se determinan las coordenadas delimitadoras de la matrícula 
                # Con estas coordenadas, la matrícula se extrae de la imagen en escala de grises
                (x, y) = np.where(mascara == 255)
                (x1, y1) = (np.min(x), np.min(y))
                (x2, y2) = (np.max(x), np.max(y))
                matriculaFinal = imagenEscalaGrises[x1:x2 + 1, y1:y2 + 1]

                # Usaremos el OCR previamente entrenado easyocr (se le pasa el idioma)       
                leerOCR = easyocr.Reader(["es"], gpu=True, verbose=False)
                textoReconocido = leerOCR.readtext(matriculaFinal)
                
                # Para mostrar por pantalla sólo la matrícula obtenida
                matriculaActual = ""
                for (puntos, matricula, probabilidad) in textoReconocido:
                    # print(f'Probabilidad: {probabilidad}')
                    # print(f'Puntos del recuadro: {puntos}')
                    # Quitamos todos los caracteres especiales (separadores, espacios y demás)
                    matricula = limpiarMatricula(matricula)                
                    matriculaActual = f"{matriculaActual}{matricula}"
                
                # Para depurar, mostramos todo lo que reconozca aunque no tenga el formato de una matrícula               
                if len(matriculaActual) > 4 and matriculaAnterior != matriculaActual:
                    print(f"Posible matrícula leída (sin validar): {matriculaActual} en frame {frameActual}")
                else:
                    matriculaActual = ""
                # Comprobamos que la matrícula leída tenga un formato válido y que no sea la anterior
                if len(matriculaActual) > 4 and validarMatricula(matriculaActual) and matriculaAnterior != matriculaActual:
                    # Para mostrar matrícula en el vídeo de la webcam en real
                    # Dejar sólo 2 decimales en la probabilidad
                    texto = "Matricula: {0} || Prob. OCR: {1:2f}".format(matriculaActual, probabilidad)
                    fuente = cv2.FONT_HERSHEY_SIMPLEX
                    tamanoFuente = 1
                    colorFuente = (255, 0, 0)  # Azul
                    grosorFuente = 4
                    # Obtener las dimensiones de la imagen
                    (h, w) = imagen.shape[:2]
                    # Calcular la posición del texto para colocarlo en la esquina inferior derecha
                    (tw, th), _ = cv2.getTextSize(texto, fuente, tamanoFuente, grosorFuente)
                    posicion = (w - tw - 10, h - 10)
                    # Colocar el texto en el vídeo
                    cv2.putText(imagen, texto, posicion, fuente, tamanoFuente, colorFuente, grosorFuente)
                    # Mostramos la matrícula también por consola
                    print(f'Matrícula leída y validada [{matriculaActual}] a las [{datetime.datetime.now()}], en frame {frameActual}')
                    # Guardamos el frame actual en fichero de imagen
                    ficheroImagen = f"{carpetaCapturas}{matriculaActual}_{frameActual}.jpg"
                    cv2.imwrite(filename=ficheroImagen, img=imagen)
                
                if len(matriculaActual) > 0:
                    matriculaAnterior = matriculaActual                                
            
            # Si queremos mostrar una ventana con la imagen en tiempo real y los contornos
            # descomentaremos esta línea
            cv2.imshow("ProyectoA - Detector de matriculas desde video", imagen)

            # Si se pulsa la tecla "s" se cierra el programa
            if cv2.waitKey(5) & 0xFF == ord('s'):
                break
        
    # Liberamos los recursos cargados de la webcam
    capturaVideo.release()
    cv2.destroyAllWindows()

else:
    print("No se ha encontrado una cámara webcam en el canal 0 o no se ha encontrado el fichero de vídeo.")