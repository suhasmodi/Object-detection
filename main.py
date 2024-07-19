import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import tkinter.font as tkFont



model = YOLO("yolov8l.pt")

def draw_boxes_and_labels(image, results):
    labels = []
    class_counts = {}

    for result in results:
        boxes = result.boxes.xyxy.numpy()  
        confidences = result.boxes.conf.numpy()  
        class_ids = result.boxes.cls.numpy()  
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = confidences[i]
            cls = int(class_ids[i])
            class_name = model.names[cls]

            
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

            
            
            
            
            
            
            

    return image, labels, class_counts

def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    image = cv2.imread(file_path)
    if image is None:
        print(f"Could not open or find the image: {file_path}")
        return

    image_resized = cv2.resize(image, (440, 380))  
    results = model(image_resized)

    output_image, labels, class_counts = draw_boxes_and_labels(image_resized.copy(), results)

    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    im_pil = Image.fromarray(output_image_rgb)
    imgtk = ImageTk.PhotoImage(image=im_pil)

    image_label.config(image=imgtk)
    image_label.image = imgtk

    
    class_summary = " ".join([f"{count} {cls}" for cls, count in class_counts.items()])
    summary_text = f"{class_summary}, {results[0].speed['inference']:.1f}ms"

    
    text_results.config(text=summary_text)


import tkinter as tk
from tkinter import filedialog, Label, Button


root = tk.Tk()
root.geometry("500x520")  
root.resizable(width=False,height=False)
root.title("YOLO Object Detection")

font = tkFont.Font(family="Helvetica", size=12, weight="bold")

upload_button = Button(root, text="Upload Image", command=upload_image,font=font,bg='blue',fg='white')
upload_button.place(x=200, y=50, height=30)

image_label = Label(root)
image_label.place(x=20, y=100,width=440, height=380)

text_results = Label(root, text="",fg='blue',font= font, justify="left")
text_results.place(x=20, y=480, width=460, height=20)

root.mainloop()
