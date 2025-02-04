import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import pickle
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import threading

# Set color and font for the User Interface
BG_COLOR = "#2b2b2b"          
ACCENT_COLOR = "#1abc9c"      
TEXT_COLOR = "#ecf0f1"        
BUTTON_BG = "#34495e"
BUTTON_FG = "#ecf0f1"
FONT_FAMILY = "Helvetica"

#Main application class
class FaceRecognitionApp:
    #to initialize the application, to set up the main window and to load the facial recognition model
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Face Recognition")
        self.root.geometry("800x650")
        self.root.configure(bg=BG_COLOR)
        
        # Initialize the InsightFace model
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        #dictonary to save registrated users' face embeddings 
        self.known_embeddings = {}
        #load registrated users' from a file
        self.load_data()
        
        #set up the styles for UI
        self.setup_styles()
        #set up the main UI
        self.setup_main_ui()

    #To set up styles for frames, buttons and labels
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TFrame", background=BG_COLOR)
        self.style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=(FONT_FAMILY, 12))
        self.style.configure("Header.TLabel", font=(FONT_FAMILY, 24, "bold"), foreground=ACCENT_COLOR, background=BG_COLOR)
        self.style.configure("TButton", font=(FONT_FAMILY, 12), padding=10, relief="flat", background=BUTTON_BG, foreground=BUTTON_FG)
        self.style.map("TButton", background=[("active", ACCENT_COLOR)], foreground=[("active", BG_COLOR)])
    
    #to create main UI with a main frame, an header and 4 buttons for different features
    def setup_main_ui(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(expand=True, fill="both", padx=40, pady=40)
        
        header = ttk.Label(self.main_frame, text="Advanced Face ID System", style="Header.TLabel")
        header.pack(pady=(0, 30))
        
        btn_style = {"width": 30}
        ttk.Button(self.main_frame, text="Register New User", command=self.register_user, **btn_style).pack(pady=10)
        ttk.Button(self.main_frame, text="Real-Time Recognition", command=self.show_real_time_ui, **btn_style).pack(pady=10)
        ttk.Button(self.main_frame, text="Non Real-Time Recognition", command=self.show_non_real_time_ui, **btn_style).pack(pady=10)
        ttk.Button(self.main_frame, text="Manage Registered Users", command=self.show_manage_users_ui, **btn_style).pack(pady=10)

    #to load face embeddings from a .pkl file
    def load_data(self):
        if os.path.exists('embeddings.pkl'):
            with open('embeddings.pkl', 'rb') as f:
                self.known_embeddings = pickle.load(f)

    #to save face embeddings to a .pkl file
    def save_data(self):
        with open('embeddings.pkl', 'wb') as f:
            pickle.dump(self.known_embeddings, f)

    #to register a new user
    def register_user(self):
        files = filedialog.askopenfilenames(title="Select 3 Face Images", filetypes=(("Image files", "*.jpg *.jpeg *.png"),))        
        
        if len(files) == 0:
            return

        if len(files) < 3:
            messagebox.showerror("Error", "Please select at least 3 images")
            return
        
        if len(files) == 0:
            return

        name = simpledialog.askstring("Input", "Enter user name:")
        if not name:
            return

        embeddings = []
        for file_path in files:
            img = cv2.imread(file_path)
            faces = self.app.get(img)
            if len(faces) != 1:
                messagebox.showerror("Error", f"Each image must contain exactly one face.\nIssue with: {os.path.basename(file_path)}")
                return
            embeddings.append(faces[0].embedding)

        #calculate the average embedding 
        avg_embedding = np.mean(embeddings, axis=0)
        #save the average embedding
        self.known_embeddings[name] = avg_embedding

        self.save_data()
        self.update_registered_list()
        messagebox.showinfo("Success", "User registered successfully!")

    # ========================================================================
    # Real-Time Recognition UI and Logic
    # ========================================================================
    def show_real_time_ui(self):
        # Hide main menu
        self.main_frame.pack_forget()
        
        # Create real-time recognition frame
        self.real_time_frame = ttk.Frame(self.root)
        self.real_time_frame.pack(expand=True, fill="both")
        
        # Add back to menu button
        ttk.Button(self.real_time_frame, text="Back to Menu", command=self.close_real_time_ui, 
                 style="TButton").pack(side="top", anchor="nw", padx=10, pady=10)
        
        # Video panel setup
        video_panel = tk.Label(self.real_time_frame, bg=BG_COLOR)
        video_panel.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        
        # Thread setup for frame processing
        self.stop_event = threading.Event()
        self.threshold = 0.6
        
        self.thread = threading.Thread(target=self.process_frame, args=(video_panel,))
        self.thread.daemon = True
        self.thread.start()

    def close_real_time_ui(self):
        # Stop the thread and release resources
        self.stop_event.set()
        if hasattr(self, 'cap'):
            self.cap.release()
        # Destroy the real-time frame
        self.real_time_frame.destroy()
        # Show main menu
        self.main_frame.pack(expand=True, fill="both", padx=40, pady=40)

    #to elaborate each webcam frames, to detec faces and to compare embeddings with stored ones.
    def process_frame(self, video_panel):
        #continues to run until the stop_event is set.
        while not self.stop_event.is_set():
            #read the webcam frame--> return ret (boolean that indicates if the reading eas successfull) + frame (the capture image)
            ret, frame = self.cap.read()
            if ret:
                #facial recognition
                faces = self.app.get(frame)
                #for each faces detected
                for face in faces:
                    #exctract coordinates of the face delimiter
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    #draw the rectangle that delimites the face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    max_similarity = 0
                    identity = "Unknown"
                    #Comparison of embeddings
                    for name, saved_embedding in self.known_embeddings.items():
                        similarity = cosine_similarity([face.embedding], [saved_embedding])[0][0]
                        if similarity > max_similarity and similarity > self.threshold:
                            max_similarity = similarity
                            identity = name

                    #If the identity is known, the color is green (0, 255, 0), otherwise it is red (0, 0, 255).
                    color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                    cv2.putText(frame, f"{identity} ({max_similarity:.2f})", (x1 + 6, y2 - 6),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
                
                #Converts the video frame from BGR (used by OpenCV) to RGB (used by PIL).
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #Corverts to a PIL image
                img_pil = Image.fromarray(img_rgb)
                #Converts to a Tkinter image (that can be displayed in the Label widget)
                imgtk = ImageTk.PhotoImage(image=img_pil)
                video_panel.imgtk = imgtk
                video_panel.configure(image=imgtk)
            #update Tkinter User Interface    
            self.root.update()

    # ========================================================================
    # Non Real-Time Recognition UI and Logic
    # ========================================================================
    def show_non_real_time_ui(self):
        # Hide main menu
        self.main_frame.pack_forget()
        
        # Create non-real-time frame
        self.nrt_frame = ttk.Frame(self.root)
        self.nrt_frame.pack(expand=True, fill="both")
        
        # Back to menu button
        ttk.Button(self.nrt_frame, text="Back to Menu", command=lambda: [self.nrt_frame.destroy(), 
                  self.main_frame.pack(expand=True, fill="both", padx=40, pady=40)], 
                 style="TButton").pack(side="top", anchor="nw", padx=10, pady=10)
        
        # Main content frame
        main_content = ttk.Frame(self.nrt_frame)
        main_content.pack(expand=True, fill="both", padx=20, pady=20)
        main_content.columnconfigure(0, weight=1)
        main_content.columnconfigure(1, weight=1)
        main_content.rowconfigure(0, weight=1)
        
        # Left frame - Image verification
        left_frame = ttk.Frame(main_content)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)
        
        ttk.Label(left_frame, text="Upload an Image to Verify", font=(FONT_FAMILY, 14)).pack(pady=10)
        
        self.nrt_image_label = tk.Label(left_frame, text="No image loaded", bg=BG_COLOR, fg=TEXT_COLOR)
        self.nrt_image_label.pack(expand=True, fill="both", padx=10, pady=10)
        
        ttk.Button(left_frame, text="Load Image", command=self.load_nrt_image, style="TButton").pack(pady=10)
        ttk.Button(left_frame, text="Verify", command=self.verify_nrt_image, style="TButton").pack(pady=10)
        
        # Right frame - Registered users
        right_frame = ttk.Frame(main_content)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        
        ttk.Label(right_frame, text="Registered Users", font=(FONT_FAMILY, 14)).pack(pady=10)
        
        self.registered_listbox = tk.Listbox(right_frame, selectmode=tk.MULTIPLE, 
                                           font=(FONT_FAMILY, 20), bg=BUTTON_BG, fg=TEXT_COLOR, selectbackground=ACCENT_COLOR)
        self.registered_listbox.pack(expand=True, fill="both", padx=10, pady=10)
        self.update_registered_list()

    #to load the image
    def load_nrt_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=(("Image files", "*.jpg *.jpeg *.png"),))
        if file_path:
            #save the path
            self.nrt_image_path = file_path
            #read the image
            img = cv2.imread(file_path)
            #conversion color: from BGR (used by OpenCV) to RGB (used by PIL).
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #create PIL image
            img_pil = Image.fromarray(img)
            #resizing
            img_pil.thumbnail((300, 300))
            #create Tkinter image (that can be displayed in the Label widget)
            self.nrt_image = ImageTk.PhotoImage(img_pil)
            #update the Label with the uploaded image (and remove the text "No image loaded")
            self.nrt_image_label.configure(image=self.nrt_image, text="")
    
    #to update the list of registered user
    def update_registered_list(self):
        #Check if the self object has an attribute called "registered_listbox"
        if hasattr(self, 'registered_listbox'):
            self.registered_listbox.delete(0, tk.END)
            if not self.known_embeddings:
                self.registered_listbox.insert(tk.END, "No registered user")
            else:
                for name in self.known_embeddings.keys():
                    self.registered_listbox.insert(tk.END, name)

    #to Check if the faces in the uploaded image match those of registered users
    def verify_nrt_image(self):
        if not self.nrt_image_path:
            messagebox.showerror("Error", "Load an image first")
            return
        
        #read the image
        img = cv2.imread(self.nrt_image_path)
        if img is None:
            messagebox.showerror("Error", "Cannot open image")
            return

        #Face detection using InsightFace model
        faces = self.app.get(img)
        if len(faces) == 0:
            messagebox.showerror("Error", "No face detected in the image")
            return

        #Definition of the similarity thresholdb
        threshold = 0.6
        recognized_faces = []
        for face in faces:
            max_similarity = 0
            identity = None
            for name, saved_embedding in self.known_embeddings.items():
                similarity = cosine_similarity([face.embedding], [saved_embedding])[0][0]
                if similarity > max_similarity and similarity > threshold:
                    max_similarity = similarity
                    identity = name
            #Storing recognised faces
            if identity:
                recognized_faces.append((face, identity, max_similarity))

        #Check if there are recognized faces in the recognized_faces list
        if recognized_faces:
            annotated_image = img.copy()
            #For each face recognized
            for face, identity, sim in recognized_faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(annotated_image)
            pil_img.thumbnail((300, 300))
            self.nrt_image = ImageTk.PhotoImage(pil_img)
            self.nrt_image_label.configure(image=self.nrt_image)

            
            #Show names of recognized faces in a messagebox
            recognized_names = ', '.join([identity for (_, identity, _) in recognized_faces])
            messagebox.showinfo("Result", f"Recognized: {recognized_names}")
        
        #Management of unrecognized faces (draw red squares around the faces detected)
        else:
            annotated_image2 = img.copy()
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(annotated_image2, (x1, y1), (x2, y2), (0, 0, 255), 15)
            annotated_image2 = cv2.cvtColor(annotated_image2, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(annotated_image2)
            pil_img.thumbnail((300, 300))
            self.nrt_image = ImageTk.PhotoImage(pil_img)
            self.nrt_image_label.configure(image=self.nrt_image)
            messagebox.showinfo("Result", "No registered face detected among those found")

    # ========================================================================
    # Manage Users UI and Logic
    # ========================================================================
    def show_manage_users_ui(self):
        # Hide main menu
        self.main_frame.pack_forget()
        
        # Create manage users frame
        self.manage_users_frame = ttk.Frame(self.root)
        self.manage_users_frame.pack(expand=True, fill="both")
        
        # Back to menu button
        ttk.Button(self.manage_users_frame, text="Back to Menu", 
                 command=lambda: [self.manage_users_frame.destroy(), 
                 self.main_frame.pack(expand=True, fill="both", padx=40, pady=40)], 
                 style="TButton").pack(side="top", anchor="nw", padx=10, pady=10)
        
        # Main content
        lbl = ttk.Label(self.manage_users_frame, text="Registered Users", font=(FONT_FAMILY, 14))
        lbl.pack(pady=10)
        
        listbox = tk.Listbox(self.manage_users_frame, selectmode=tk.MULTIPLE, 
                           font=(FONT_FAMILY, 20), bg=BUTTON_BG, fg=TEXT_COLOR, 
                           selectbackground=ACCENT_COLOR)
        listbox.pack(expand=True, fill="both", padx=10, pady=10)
        
        self.registered_listbox = listbox
        
        # Population of the user list
        for name in self.known_embeddings.keys():
            listbox.insert(tk.END, name)
        
        # Delete selected users functionality
        def delete_selected():
            selected_indices = listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Warning", "Select at least one user to delete")
                return
            
            for index in reversed(selected_indices):
                name = listbox.get(index).strip()
                if name in self.known_embeddings:
                    del self.known_embeddings[name]
                    listbox.delete(index)
            self.save_data()
            self.update_registered_list()
            messagebox.showinfo("Success", "Selected users have been deleted!")
        
        ttk.Button(self.manage_users_frame, text="Delete Selected Users", 
                 command=delete_selected, style="TButton").pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
