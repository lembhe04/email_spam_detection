# app.py - UPDATED VERSION
import tkinter as tk
from tkinter import scrolledtext, messagebox
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from predict import predict_email
except ImportError as e:
    messagebox.showerror("Error", f"Failed to import dependencies: {str(e)}")
    sys.exit(1)

class SpamDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Email Spam Detector")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root)
        title_frame.pack(pady=10)
        
        tk.Label(
            title_frame,
            text="Email Spam Detector",
            font=("Helvetica", 18, "bold")
        ).pack()
        
        tk.Label(
            title_frame,
            text="Paste your email content below to check if it's spam",
            font=("Helvetica", 10)
        ).pack(pady=5)
        
        # Text input
        tk.Label(
            self.root,
            text="Email Content:",
            font=("Helvetica", 10),
            anchor="w"
        ).pack(fill="x", padx=20, pady=(10, 0))
        
        self.email_text = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            width=70,
            height=15,
            font=("Helvetica", 10)
        )
        self.email_text.pack(padx=20, pady=5)
        
        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        tk.Button(
            button_frame,
            text="Detect Spam",
            command=self.detect_spam,
            bg="#4CAF50",
            fg="white",
            font=("Helvetica", 10, "bold"),
            width=15
        ).pack(side="left", padx=5)
        
        tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_text,
            bg="#f44336",
            fg="white",
            font=("Helvetica", 10, "bold"),
            width=15
        ).pack(side="left", padx=5)
        
        # Result
        self.result_frame = tk.Frame(self.root, bd=2, relief="groove")
        self.result_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Label(
            self.result_frame,
            text="Result:",
            font=("Helvetica", 10, "bold"),
            anchor="w"
        ).pack(fill="x", padx=10, pady=(10, 0))
        
        self.result_label = tk.Label(
            self.result_frame,
            text="",
            font=("Helvetica", 12, "bold"),
            pady=20
        )
        self.result_label.pack(fill="both", expand=True)
    
    def detect_spam(self):
        email_content = self.email_text.get("1.0", tk.END).strip()
        
        if not email_content:
            messagebox.showwarning("Warning", "Please enter email content to analyze!")
            return
        
        try:
            result = predict_email(email_content)
            self.result_label.config(text=result)
            
            if result == "Spam":
                self.result_label.config(bg="#ffcdd2", fg="#d32f2f")  # Red
            else:
                self.result_label.config(bg="#c8e6c9", fg="#388e3c")  # Green
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def clear_text(self):
        self.email_text.delete("1.0", tk.END)
        self.result_label.config(text="", bg=self.result_frame.cget("bg"))

if __name__ == "__main__":
    root = tk.Tk()
    app = SpamDetectorApp(root)
    root.mainloop()