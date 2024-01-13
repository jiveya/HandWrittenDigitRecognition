#-------------------------------------------------------------------------------
# Name:        module2
# Purpose:
#
# Author:      60111
#
# Created:     03/01/2024
# Copyright:   (c) 60111 2024
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import cv2
import glob
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageDraw, ImageGrab

# Load trained model
model = joblib.load('model_knn3.pkl')

def show_image(img):
    # Assuming 'scaled_img' contains the processed image
    plt.imshow(img, cmap='gray')
    plt.colorbar()  # Optional - adds a colorbar to show value range
    plt.show()

def preprocess_image(file):
    # Read the image, preprocess, and prepare for prediction
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 28x28 with the aspect ratio
    img = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
    show_image(img)

    # Convert to 1D array
    img = img.reshape(-1)

    # Invert the color and normalized to 0 and 1
    img = (255.0 - img) / 255.0
    img = np.array([img]).astype(np.float32)
    show_image(img.reshape(28,28))

    return img

def Recognize_Digit():
    global model
    # Grab the drawing from the canvas
    x = root.winfo_rootx() + cv.winfo_x()
    y = root.winfo_rooty() + cv.winfo_y()
    x1 = x + cv.winfo_width()
    y1 = y + cv.winfo_height()

    # Save the canvas content as an image
    filename = "images.png"
    ImageGrab.grab(bbox=(x+70, y+70, x1+50, y1+50)).save(filename)

    img = preprocess_image(filename)

    # Make prediction using the KNN model
    res = model.predict(img)[0]
    print(res)

    # Show the predicted digit in a message box
    messagebox.showinfo("Prediction", f"The predicted digit is: {res}")

def clear_widget():
    global cv
    #To clear a canvas
    cv.delete("all")

def activate_event(event):
    global lastx, lasty
    # <B1-Motion>
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    # do the canvas drawings
    cv.create_line((lastx, lasty, x, y),width=18, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y

def upload_image():
    file_path = filedialog.askopenfilename(initialdir="/", title="Select Image File", filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")))

##    filename = filedialog.askopenfile()
    img = preprocess_image(file_path)

    # Make prediction using the KNN model
    res = model.predict(img)[0]
    print(res)

    # Show the predicted digit in a message box
    messagebox.showinfo("Prediction", f"The predicted digit is: {res}")


# Create Tkinter window
root = tk.Tk()
root.resizable(0,0)
root.title("Digit Recognition using KNN")

# Initialize variables
lastx, lasty = None, None
image_number = 0

nb = ttk.Notebook(root)

# Frame for drawing
frame_draw = ttk.Frame(nb)
frame_draw.pack(fill= tk.BOTH, expand=True)
nb.add(frame_draw, text="Draw")

# Create a canvas for drawing
cv = Canvas(frame_draw, width=540, height=380, bg='white')
cv.pack()
cv.bind('<Button-1>', activate_event)

# Button for drawing and recognize
btn_recognize = Button(frame_draw, text="Recognize Digit", command=Recognize_Digit)
btn_recognize.pack(padx = 10, pady = 10, side = tk.LEFT)

# Button for clearing the drawing
button_clear = Button(frame_draw, text = "Clear Widget", command=clear_widget)
button_clear.pack(padx = 10, pady = 10, side = tk.RIGHT)

# Frame for uploading image
frame_upload = ttk.Frame(nb)
frame_upload.pack(fill= tk.BOTH, expand=True)
nb.insert("end", frame_upload, text = "Upload")
nb.pack(padx = 5, pady = 5, expand = True)

# Button for uploading image and recognize
btn_upload = Button(frame_upload, text="Upload Image", command=upload_image)
btn_upload.pack(pady = 50, padx = 20)

root.mainloop()

