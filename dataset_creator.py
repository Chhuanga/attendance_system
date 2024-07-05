import cv2
import sqlite3
import tkinter as tk
from tkinter import messagebox
import os

os.environ["THEANO_FLAG"]="device=cuda,assert_no_cpu_op = True"

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

def insert_record(first_name, last_name, roll_no, dept):
    conn = sqlite3.connect("sqlite1.db")
    conn.execute("INSERT INTO students (first_name, last_name, roll_no, dept) VALUES (?,?,?,?)",(first_name, last_name, roll_no, dept))
    conn.commit()
    conn.close()

def get_file_name(first_name, roll_no, count):
    if first_name and roll_no:
        return f"{first_name}.{roll_no}.{count}"
    else:
        return None

def get_student_details():
    details = {}
    details_saved = False  # Flag to track whether details are saved

    root = tk.Tk()
    root.title("Enter Student Details")

    # Calculate the center position of the screen
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 400
    window_height = 300
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    # Set the window size and position
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def on_save():
        nonlocal details_saved
        roll_no = roll_entry.get()
        if not roll_no:
            messagebox.showerror("Error", "Please enter a Roll Number.")
            return

        # Check if roll_no already exists in the database
        conn = sqlite3.connect("sqlite1.db")
        cursor = conn.execute("SELECT roll_no FROM students WHERE roll_no=?", (roll_no,))
        existing_roll_no = cursor.fetchone()
        conn.close()

        if existing_roll_no:
            messagebox.showerror("Error", f"Roll Number '{roll_no}' already exists.")
            return

        details["first_name"] = first_name_entry.get()
        details["last_name"] = last_name_entry.get()
        details["roll_no"] = roll_no
        details["dept"] = dept_entry.get()
        insert_record(details["first_name"], details["last_name"], details["roll_no"], details["dept"])
        messagebox.showinfo("Success", "Student Information Saved!")
        details_saved = True
        root.destroy()
    def on_close():
        nonlocal details_saved
        if any(entry.get() != "" for entry in (first_name_entry, last_name_entry, roll_entry, dept_entry)):
            if not messagebox.askokcancel("Save", "Do you want to save the changes?"):
                return
            on_save()
        else:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    first_name_label = tk.Label(root, text="First Name:")
    first_name_label.pack()
    first_name_entry = tk.Entry(root)
    first_name_entry.pack()

    last_name_label = tk.Label(root, text="Last Name:")
    last_name_label.pack()
    last_name_entry = tk.Entry(root)
    last_name_entry.pack()

    roll_label = tk.Label(root, text="Roll Number:")
    roll_label.pack()
    roll_entry = tk.Entry(root)
    roll_entry.pack()

    dept_label = tk.Label(root, text="Department:")
    dept_label.pack()
    dept_entry = tk.Entry(root)
    dept_entry.pack()

    save_button = tk.Button(root, text="Save", command=on_save)
    save_button.pack()

    root.mainloop()

    if not details_saved:
        return None, None

    return details.get("first_name"), details.get("roll_no")

first_name, roll_no = get_student_details()

if first_name is not None and roll_no is not None:

    sampleNum = 0
    totalImages = 500

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sampleNum += 1
            progress_percentage = int((sampleNum / totalImages) * 100)
            file_name = get_file_name(first_name, roll_no, sampleNum)
            if file_name is not None:
                print(f"Saving image as: {file_name}.jpg")
                cv2.imwrite(f"dataset/{file_name}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            progress_text = f"{progress_percentage}%"
            cv2.putText(img, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Face", img)
        if sampleNum >= totalImages:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
