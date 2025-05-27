# Libraries
from tkinter import *
from PIL import Image, ImageTk
import imutils
import cv2
import numpy as np
from ultralytics import YOLO
import math
import platform

# Función para verificar carga de imágenes
def check_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Error cargando imagen: {path}")
    return img

def clean_lbl():
    lblimg.config(image='')
    lblimgtxt.config(image='')

def images(img, imgtxt):
    img = np.array(img, dtype="uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    img_ = ImageTk.PhotoImage(image=img)
    lblimg.configure(image=img_)
    lblimg.image = img_

    imgtxt = np.array(imgtxt, dtype="uint8")
    imgtxt = cv2.cvtColor(imgtxt, cv2.COLOR_BGR2RGB)
    imgtxt = Image.fromarray(imgtxt)
    img_txt = ImageTk.PhotoImage(image=imgtxt)
    lblimgtxt.configure(image=img_txt)
    lblimgtxt.image = img_txt

def Scanning():
    global img_metal, img_glass, img_plastic, img_carton, img_medical
    global img_metaltxt, img_glasstxt, img_plastictxt, img_cartontxt, img_medicaltxt, pantalla
    global lblimg, lblimgtxt

    lblimg = Label(pantalla)
    lblimg.place(x=75, y=260)

    lblimgtxt = Label(pantalla)
    lblimgtxt.place(x=995, y=310)
    detect = False

    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("❌ No se pudo capturar un frame de la cámara")
            clean_lbl()
            lblVideo.after(1000, Scanning)
            return

        frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame, stream=True, verbose=False)

        for res in results:
            boxes = res.boxes
            for box in boxes:
                detect = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
                cls = int(box.cls[0])
                conf = math.ceil(box.conf[0])
                text = f'{clsName[cls]} {int(conf) * 100}%'
                size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                dim, baseline = size[0], size[1]

                colors = [(255,255,0), (255,255,255), (0,0,255), (150,150,150), (255,0,0)]
                imgs = [img_metal, img_glass, img_plastic, img_carton, img_medical]
                txts = [img_metaltxt, img_glasstxt, img_plastictxt, img_cartontxt, img_medicaltxt]

                cv2.rectangle(frame_show, (x1, y1), (x2, y2), colors[cls], 2)
                cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[cls], 2)

                images(imgs[cls], txts[cls])

        if not detect:
            clean_lbl()

        frame_show = imutils.resize(frame_show, width=640)
        im = Image.fromarray(frame_show)
        img = ImageTk.PhotoImage(image=im)
        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(10, Scanning)
    else:
        print("❌ No se pudo abrir la cámara")
        clean_lbl()

def ventana_principal():
    global cap, lblVideo, model, clsName, img_metal, img_glass, img_plastic, img_carton, img_medical
    global img_metaltxt, img_glasstxt, img_plastictxt, img_cartontxt, img_medicaltxt, pantalla

    pantalla = Tk()
    pantalla.title("RECICLAJE INTELIGENTE")
    pantalla.geometry("1280x720")

    try:
        imagenF = PhotoImage(file="setUp/Canva.png")
        background = Label(image=imagenF, text="Inicio")
        background.place(x=0, y=0, relwidth=1, relheight=1)
    except Exception as e:
        print(f"⚠️ Error cargando fondo: {e}")

    model = YOLO('Modelos/best.pt')
    clsName = ['Metal', 'Glass', 'Plastic', 'Carton', 'Medical']

    # Cargar imágenes
    img_metal = check_image("setUp/metal.png")
    img_glass = check_image("setUp/vidrio.png")
    img_plastic = check_image("setUp/plastico.png")
    img_carton = check_image("setUp/carton.png")
    img_medical = check_image("setUp/medical.png")
    img_metaltxt = check_image("setUp/metaltxt.png")
    img_glasstxt = check_image("setUp/vidriotxt.png")
    img_plastictxt = check_image("setUp/plasticotxt.png")
    img_cartontxt = check_image("setUp/cartontxt.png")
    img_medicaltxt = check_image("setUp/medicaltxt.png")

    lblVideo = Label(pantalla)
    lblVideo.place(x=320, y=180)

    # Iniciar cámara
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)

    if cap.isOpened():
        cap.set(3, 1280)
        cap.set(4, 720)
    else:
        print("❌ No se pudo acceder a la cámara.")

    Scanning()
    pantalla.mainloop()

if __name__ == "__main__":
    ventana_principal()
