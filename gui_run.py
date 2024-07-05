import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import sqlite3
import subprocess
from datetime import datetime
import pandas as pd

os.environ["THEANO_FLAG"]="device=cuda,assert_no_cpu_op = True"
def on_closing(window):
    window.destroy()
    root.deiconify()  # It will reopen the attendance system window when other windows are closed

def get_images_with_id(path, progress_var, progress_bar):
    images_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    total_images = len(images_paths)
    for idx, single_image_path in enumerate(images_paths):
        faceImg = Image.open(single_image_path).convert('L')
        faceNp = np.array(faceImg, np.uint8)
        id = int(os.path.split(single_image_path)[-1].split(".")[1])
        faces.append(faceNp)
        ids.append(id)
        progress = int((idx + 1) / total_images * 100)
        progress_var.set(progress)
        progress_bar.update()
        cv2.waitKey(10)

    return np.array(ids), faces

def gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, table)

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = "dataset"

    progress_window = tk.Toplevel(root)
    progress_window.title("Training Progress")
    progress_window.geometry("300x50")

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate', variable=progress_var)
    progress_bar.pack(pady=10)

    ids, faces = get_images_with_id(path, progress_var, progress_bar)
    recognizer.train(faces, ids)
    recognizer.save("recog/trainingdata.yml")
    cv2.destroyAllWindows()
    messagebox.showinfo("Train Model", "Training Complete")
    progress_window.after(100, progress_window.destroy)

def open_dataset_creator():
    try:
        process = subprocess.Popen(["python", "dataset_creator.py"])
        process.wait()
        messagebox.showinfo("Process Complete", "Dataset creation process completed successfully.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Failed to run dataset_creator.py: {e}")

def manual_attendance():
    def record_attendance():
        roll_no = roll_no_entry.get()
        c_id = c_id_entry.get()

        if not roll_no or not c_id:
            messagebox.showerror("Error", "Please enter Roll No and Subject Code")
            return

        conn = sqlite3.connect("sqlite1.db")
        cursor = conn.cursor()

        cursor.execute("SELECT s_id FROM students WHERE roll_no=?", (roll_no,))
        student_info = cursor.fetchone()

        if student_info is None:
            messagebox.showerror("Error", "Student not found")
            conn.close()
            return

        s_id = student_info[0]
        current_date = datetime.now().strftime("%Y-%m-%d")

        cursor.execute("INSERT INTO attendance (s_id, c_id, attend_date) VALUES (?, ?, ?)", (s_id, c_id, current_date))
        conn.commit()

        conn.close()
        messagebox.showinfo("Success", "Attendance recorded successfully")


    manual_attendance_window = tk.Toplevel(root)
    manual_attendance_window.title("Manual Attendance")

    # Calculating the position to center the window
    window_width = 300
    window_height = 150
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2


    manual_attendance_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    roll_no_label = tk.Label(manual_attendance_window, text="Roll No:")
    roll_no_label.pack()
    roll_no_entry = tk.Entry(manual_attendance_window)
    roll_no_entry.pack()

    c_id_label = tk.Label(manual_attendance_window, text="Subject Code:")
    c_id_label.pack()
    c_id_entry = tk.Entry(manual_attendance_window)
    c_id_entry.pack()

    record_button = tk.Button(manual_attendance_window, text="Record Attendance", command=record_attendance)
    record_button.pack()

def generate_monthly_attendance():
    def calculate_attendance_percentage(s_id, month):
        conn = sqlite3.connect("sqlite1.db")
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COUNT(DISTINCT attend_date) FROM attendance WHERE s_id=? AND strftime('%Y-%m', attend_date)=?",
            (s_id, month))
        total_classes = cursor.fetchone()[0]

        # Counting attended classes in the month
        cursor.execute(
            "SELECT COUNT(DISTINCT attend_date) FROM attendance WHERE s_id=? AND strftime('%Y-%m', attend_date)=? AND c_id IN (SELECT c_id FROM class_routine WHERE Day=strftime('%A', attend_date) AND start_time<=strftime('%H:%M:%S', attend_date) AND end_time>=strftime('%H:%M:%S', attend_date))",
            (s_id, month))
        attended_classes = cursor.fetchone()[0]

        conn.close()

        if total_classes == 0:
            return 0.0
        return (attended_classes / total_classes) * 100

    def get_class_name(c_id):
        conn = sqlite3.connect("sqlite1.db")
        cursor = conn.cursor()

        cursor.execute("SELECT class_name FROM class WHERE c_id=?", (c_id,))
        class_name = cursor.fetchone()[0]

        conn.close()

        return class_name

    def store_monthly_attendance(s_id, month, attendance_percentage):
        conn = sqlite3.connect("sqlite1.db")
        cursor = conn.cursor()

        cursor.execute("INSERT INTO monthly_attend (s_id, Month, attend_percentage) VALUES (?, ?, ?)", (s_id, month, attendance_percentage))
        conn.commit()

        conn.close()

    def display_attendance_details(month):
        conn = sqlite3.connect("sqlite1.db")
        cursor = conn.cursor()

        cursor.execute("SELECT s_id, attend_percentage FROM monthly_attend WHERE Month=?", (month,))
        monthly_attendance = cursor.fetchall()

        detail_window = tk.Toplevel(root)
        detail_window.title("Monthly Attendance Details")


        cursor.execute("SELECT DISTINCT class_name FROM class")
        class_names = [class_name[0] for class_name in cursor.fetchall()]

        # Creating columns for treeview
        columns = ("Roll No", "Name") + tuple(class_names) + ("Total Percentage",)
        tree = ttk.Treeview(detail_window, columns=columns)
        tree.heading("#0", text="Student ID")
        tree.column("#0", minwidth=0, width=100)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, minwidth=0, width=100)

        # Fetching student details
        cursor.execute("SELECT s_id, first_name, last_name, roll_no FROM students ORDER BY roll_no ASC")
        student_data = cursor.fetchall()

        class_name_to_index = {name: i for i, name in enumerate(class_names)}

        for s_id, first_name, last_name, roll_no in student_data:
            total_classes_attended = [0] * len(class_names)
            cursor.execute("SELECT c_id, attend_date FROM attendance WHERE s_id=?", (s_id,))
            all_attendance_records = cursor.fetchall()

            print("All Attendance Records for student", s_id, ":", all_attendance_records)

            for c_id, attend_date in all_attendance_records:
                if attend_date.startswith(month + '-'):
                    class_name = get_class_name(c_id)
                    class_index = class_names.index(class_name)
                    total_classes_attended[class_index] += 1

            print("Total Classes Attended for student", s_id, ":", total_classes_attended)

            total_classes = sum(total_classes_attended)
            attendance_percentage = (total_classes / len(all_attendance_records)) * 100 if len(
                all_attendance_records) > 0 else 0

            print("Attendance Percentage for", s_id, ":", attendance_percentage)

            tree.insert("", "end", text=s_id,
                        values=(roll_no, f"{first_name} {last_name}",) + tuple(total_classes_attended) + (
                            attendance_percentage,))

        tree.pack()

        conn.close()

    def export_to_excel(month):
        conn = sqlite3.connect("sqlite1.db")
        cursor = conn.cursor()

        cursor.execute("SELECT s_id, attend_percentage FROM monthly_attend WHERE Month=?", (month,))
        monthly_attendance = cursor.fetchall()

        student_names = []
        attendance_percentages = []
        for s_id, percentage in monthly_attendance:
            cursor.execute("SELECT first_name, last_name FROM students WHERE s_id=?", (s_id,))
            student_name = cursor.fetchone()
            if student_name:
                name = f"{student_name[0]} {student_name[1]}"
                student_names.append(name)
                attendance_percentages.append(percentage)

        cursor.execute("SELECT DISTINCT class_name FROM class")
        class_names = [class_name[0] for class_name in cursor.fetchall()]

        df = pd.DataFrame(columns=["Roll No", "Name"] + class_names + ["Total Percentage"])
        df["Roll No"] = [roll_no for roll_no, _ in monthly_attendance]
        df["Name"] = student_names

        for i, (_, percentage) in enumerate(monthly_attendance):
            df.at[i, "Total Percentage"] = percentage
            cursor.execute("SELECT c_id FROM attendance WHERE s_id=?", (df.at[i, "Roll No"],))
            attendance_records = cursor.fetchall()
            for c_id, _ in attendance_records:
                class_name = cursor.execute("SELECT class_name FROM class WHERE c_id=?", (c_id,)).fetchone()[0]
                df.at[i, class_name] = df.at[i, class_name] + 1 if class_name in df.columns else 1

        file_name = f"Monthly_Attendance_{month}.xlsx"
        df.to_excel(file_name, index=False)
        messagebox.showinfo("Export Successful", f"Attendance details exported to {file_name}")

        conn.close()

    def center_window(window):
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def on_ok_click():
        selected_month = month_var.get()
        if not selected_month:
            messagebox.showerror("Error", "Please select a month")
            return

        conn = sqlite3.connect("sqlite1.db")
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT s_id FROM attendance")
        student_ids = cursor.fetchall()

        for student_id in student_ids:
            s_id = student_id[0]
            cursor.execute("SELECT * FROM monthly_attend WHERE s_id=? AND Month=?", (s_id, selected_month))
            if not cursor.fetchone():
                attendance_percentage = calculate_attendance_percentage(s_id, selected_month)
                store_monthly_attendance(s_id, selected_month, attendance_percentage)

        conn.close()

        display_attendance_details(selected_month)
        export_to_excel(selected_month)

    monthly_attendance_window = tk.Toplevel(root)
    monthly_attendance_window.title("Generate Monthly Attendance")
    center_window(monthly_attendance_window)

    month_label = tk.Label(monthly_attendance_window, text="Select Month:")
    month_label.pack()
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    month_var = tk.StringVar()
    month_dropdown = ttk.Combobox(monthly_attendance_window, textvariable=month_var, values=months)
    month_dropdown.pack()

    ok_button = tk.Button(monthly_attendance_window, text="OK", command=on_ok_click)
    ok_button.pack()

def display_all_students():
    # Connect to the database
    conn = sqlite3.connect("sqlite1.db")
    cursor = conn.cursor()

    # Fetch all student records
    cursor.execute("SELECT roll_no, first_name, last_name FROM students")
    student_records = cursor.fetchall()

    # Close the database connection
    conn.close()

    # Create a new window to display student details
    student_window = tk.Toplevel(root)
    student_window.title("All Students")

    # Calculate the window position for centering it on the screen
    window_width = 400
    window_height = 400
    screen_width = student_window.winfo_screenwidth()
    screen_height = student_window.winfo_screenheight()
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2
    student_window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    # Create a treeview widget to display the student details
    tree = ttk.Treeview(student_window)
    tree["columns"] = ("Name",)
    tree.heading("#0", text="Roll Number")
    tree.column("#0", minwidth=0, width=100)
    tree.heading("Name", text="Name")

    # Insert the student records into the treeview
    for record in student_records:
        student_name = record[1] + " " + record[2]
        tree.insert("", "end", text=record[0], values=(student_name,))

    # Pack the treeview widget
    tree.pack()
def open_admin_functions_window():
    def on_closing():
        root.deiconify()
        admin_window.destroy()

    admin_window = tk.Toplevel(root)
    admin_window.title("Admin Functions")

    # Calculating the center position of the screen
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 500
    window_height = 400
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    admin_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    register_student_button = tk.Button(admin_window, text="Register Student", command=open_dataset_creator)
    register_student_button.pack(pady=20)

    train_model_button = tk.Button(admin_window, text="Train Model",command=train_model)
    train_model_button.pack(pady=20)

    manual_attendance_button = tk.Button(admin_window, text="Manual Attendance",command=manual_attendance)
    manual_attendance_button.pack(pady=20)

    generate_report_button = tk.Button(admin_window, text="Generate Report", command=generate_monthly_attendance)
    generate_report_button.pack(pady=20)

    all_student_button = tk.Button(admin_window, text="Students", command=display_all_students)
    all_student_button.pack(pady=20)

    admin_window.protocol("WM_DELETE_WINDOW", on_closing)

    root.withdraw()
    admin_window.mainloop()

def open_admin_window():
    root.withdraw()
    admin_window = tk.Toplevel(root)
    admin_window.title("Admin")
    admin_window.geometry("800x600")

    screen_width = admin_window.winfo_screenwidth()
    screen_height = admin_window.winfo_screenheight()
    window_width = 800
    window_height = 600
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2


    admin_window.geometry(f"800x600+{x}+{y}")

    admin_window.protocol("WM_DELETE_WINDOW", lambda: on_closing(admin_window))


    header_frame = tk.Frame(admin_window, bd=2, relief="solid")
    header_frame.pack(side="top", fill="x", pady=10)

    # Logo and college information
    logo_image = tk.PhotoImage(file='Logo.png')
    resized_logo = logo_image.subsample(4, 4)
    logo_label = tk.Label(header_frame, image=resized_logo)
    logo_label.pack(side="left", padx=10, pady=10)

    college_label = tk.Label(header_frame, text="St.Edmund's College, Department of Computer Application", font=("Arial", 16))
    college_label.pack(side="left", padx=10, pady=10)

    login_frame = tk.Frame(admin_window)
    login_frame.pack(pady=20)

    default_username = tk.StringVar(value="")
    username_label = tk.Label(login_frame, text="Username:")
    username_label.pack()

    username_entry = tk.Entry(login_frame, textvariable=default_username)
    username_entry.pack()

    default_password = tk.StringVar(value="")
    password_label = tk.Label(login_frame, text="Password:")
    password_label.pack()

    password_entry = tk.Entry(login_frame, show="*", textvariable=default_password)
    password_entry.pack()

    conn = sqlite3.connect('sqlite1.db')
    cursor = conn.cursor()

    def admin_login():
        # Gets the username and password from the entry widgets
        username = username_entry.get()
        password = password_entry.get()

        query = "SELECT * FROM admin_credentials WHERE username=? AND password=?"
        cursor.execute(query, (username, password))
        result = cursor.fetchone()

        if result:
            messagebox.showinfo("Admin Login", "Login Successful")
            conn.close()
            admin_window.destroy()  # Close the login window
            open_admin_functions_window()  # Open the admin functions window
        else:
            messagebox.showerror("Admin Login", "Invalid Credentials")

    login_button = tk.Button(login_frame, text="Login", command=admin_login)
    login_button.pack()







def open_take_attendance_window():
    os.environ["THEANO_FLAG"] = "device=cuda,assert_no_cpu_op=True"
    root.withdraw()
    cam = cv2.VideoCapture(0)

    facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("recog/trainingdata.yml")

    def get_profile(roll_no):
        conn = sqlite3.connect("sqlite1.db")
        cursor = conn.execute("SELECT * FROM students WHERE roll_no=?", (roll_no,))
        profile = None
        for row in cursor:
            profile = row
        conn.close()
        return profile

    def display_profile(img, x, y, h, profile, marked, class_exists, conf):
        first_name = profile[1]
        last_name = profile[2]
        roll_no = profile[3]

        cv2.putText(img, f"Name: {first_name} {last_name}", (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
        cv2.putText(img, f"Roll No: {roll_no}", (x, y + h + 45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
        cv2.putText(img, f"Confidence: {conf:.2f}", (x, y + h + 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
        if marked:
            cv2.putText(img, "Attendance Marked", (x, y + h + 95), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        elif not class_exists:
            cv2.putText(img, "No class", (x, y + h + 95), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    def update_attendance(roll_no):
        conn = sqlite3.connect("sqlite1.db")
        cursor = conn.cursor()

        # Gets current day and time
        now = datetime.now()
        current_day = now.strftime("%A")
        current_time = now.strftime("%H:%M")

        print(f"Current Day: {current_day}, Current Time: {current_time}")

        # Debug print: Display all records in class_routine table for the current day
        cursor.execute("SELECT * FROM class_routine WHERE Day = ?", (current_day,))
        routine_for_day = cursor.fetchall()
        print(f"Class Routine for {current_day}: {routine_for_day}")

        # Iterate over routine_for_day to debug individual entries
        for routine in routine_for_day:
            print(f"Class ID: {routine[1]}, Start Time: {routine[3]}, End Time: {routine[4]}")
            if routine[3] <= current_time <= routine[4]:
                print(f"Matching class found: {routine}")
            else:
                print(f"No match for this class: {routine}")

        # Finds the class that matches the current day and time
        cursor.execute("SELECT * FROM class_routine WHERE Day = ? AND start_time <= ? AND end_time >= ?",
                       (current_day, current_time, current_time))
        class_info = cursor.fetchone()
        print(f"Class Info: {class_info}")

        if class_info is not None:
            c_id = class_info[1]
            attend_date = now.strftime("%Y-%m-%d")
            s_id = None

            # Get the s_id using the roll_no
            cursor.execute("SELECT s_id FROM students WHERE roll_no=?", (roll_no,))
            student_info = cursor.fetchone()
            print(f"Student Info: {student_info}")

            if student_info is not None:
                s_id = student_info[0]

            # Insert the attendance record
            if s_id is not None:
                cursor.execute("SELECT * FROM attendance WHERE s_id = ? AND c_id = ? AND attend_date = ?",
                               (s_id, c_id, attend_date))
                existing_attendance = cursor.fetchone()
                print(f"Existing Attendance: {existing_attendance}")

                if existing_attendance is None:
                    cursor.execute("INSERT INTO attendance (s_id, c_id, attend_date) VALUES (?, ?, ?)",
                                   (s_id, c_id, attend_date))
                    conn.commit()
                    marked = True
                else:
                    marked = False
            else:
                marked = False
        else:
            marked = False

        conn.close()
        return marked, class_info is not None

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        class_exists = False
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box when face is detected
            id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            profile = get_profile(id)
            if profile is not None and conf < 65:
                marked, class_exists = update_attendance(profile[3])
                display_profile(img, x, y, h, profile, marked, class_exists, conf)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box when face is recognized
            else:
                cv2.putText(img, "Unknown", (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("FACE", img)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    root.deiconify()

def show_attendance_details():
    # Connect to the database
    conn = sqlite3.connect("sqlite1.db")
    cursor = conn.cursor()

    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Fetch attendance records for today
    cursor.execute(
        "SELECT attendance.s_id, students.roll_no, students.first_name, students.last_name, attendance.c_id, attendance.attend_date FROM attendance INNER JOIN students ON attendance.s_id = students.s_id WHERE attend_date=?",
        (current_date,))
    attendance_records = cursor.fetchall()

    # Close the database connection
    conn.close()

    # Create a new window to display attendance details
    detail_window = tk.Toplevel(root)
    detail_window.title("Today's Attendance")

    # Create a treeview widget to display the attendance details
    tree = ttk.Treeview(detail_window)
    tree["columns"] = ("Roll Number", "Student Name", "Class ID", "Attendance Date")
    tree.heading("#0", text="Student ID")
    tree.column("#0", minwidth=0, width=100)
    tree.heading("Roll Number", text="Roll Number")
    tree.heading("Student Name", text="Student Name")
    tree.heading("Class ID", text="Class ID")
    tree.heading("Attendance Date", text="Attendance Date")

    # Insert the attendance records into the treeview
    for record in attendance_records:
        student_name = record[2] + " " + record[3]
        tree.insert("", "end", text=record[0], values=(record[1], student_name, record[4], record[5]))

    # Pack the treeview widget
    tree.pack()

def display_current_day_routine():

    conn = sqlite3.connect('D:/FaceAttendance/sqlite1.db')
    cursor = conn.cursor()

    # Get the current day of the week (Monday=0, Sunday=6)
    current_day = datetime.now().strftime('%A')

    # Retrieves the routine for the current day
    cursor.execute("SELECT class_routine.c_id, class_routine.start_time, class_routine.end_time, teacher_details.name FROM class_routine JOIN teacher_details ON class_routine.t_id = teacher_details.t_id WHERE class_routine.day=?", (current_day,))
    routine = cursor.fetchall()


    root = Tk()
    root.title(f"Today's Routine ({current_day})")
    root.geometry("850x300")

    # Create a treeview widget
    tree = ttk.Treeview(root, columns=("Class ID", "Start Time", "End Time", "Teacher Name"), show="headings")
    tree.pack(fill=BOTH, expand=YES)

    tree.heading("Class ID", text="Class ID")
    tree.heading("Start Time", text="Start Time")
    tree.heading("End Time", text="End Time")
    tree.heading("Teacher Name", text="Teacher Name")

    for row in routine:
        tree.insert("", "end", values=(row[0], row[1], row[2], row[3]))

    # Center the window on the screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    root.mainloop()
    conn.close()




root = tk.Tk()
root.title("FaceLink")


screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = 800
window_height = 600
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.resizable(True, True)

header_frame = tk.Frame(root, bd=2, relief="solid")
header_frame.pack(side="top", fill="x", pady=10)

# Logo and college information
logo_image = tk.PhotoImage(file='Logo_031951.png')
resized_logo = logo_image.subsample(4, 4)
logo_label = tk.Label(header_frame, image=resized_logo)
logo_label.pack(side="left", padx=10, pady=10)

college_label = tk.Label(header_frame, text="St.Edmund's College, Department of Computer Application", font=("Arial", 16))
college_label.pack(side="left", padx=10, pady=10)

take_attendance_button = tk.Button(root, text="Take Attendance", width=20, height=2, command=open_take_attendance_window)
take_attendance_button.pack()

admin_button = tk.Button(root, text="Admin", width=20, height=2,command=open_admin_window)
admin_button.pack(pady=50)

today_button = tk.Button(root, text="Today's Attendance", width=20, height=2,command=show_attendance_details)
today_button.pack()

today_routine = tk.Button(root, text="Today's Routine", width=20, height=2,command=display_current_day_routine)
today_routine.pack(pady=50)

root.mainloop()