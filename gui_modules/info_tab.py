# src/gui_modules/info_tab.py
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont

class InfoTab(ttk.Frame):
    """
    Educational Tab: Neural Networks & PINNs for Beginners.
    Styled with a modern 'Card' layout for better readability.
    """
    def __init__(self, parent, shared_state):
        super().__init__(parent)
        self.shared_state = shared_state
        
        # --- Design Palette ---
        self.colors = {
            "bg_main": "#F0F2F5",       # Soft Gray background
            "card_bg": "#FFFFFF",       # White cards
            "text_header": "#2C3E50",   # Dark Blue-Grey
            "text_body": "#4A4A4A",     # Soft Black
            "accent": "#3498DB",        # Bright Blue
            "highlight": "#E8F6F3",     # Light Teal for highlights
            "code_bg": "#2E3440",       # Dark for code blocks
            "code_fg": "#D8DEE9"        # Light text for code
        }
        
        self._init_fonts()
        self._create_content()

    def _init_fonts(self):
        """Initialize custom fonts for a cleaner look."""
        self.fonts = {
            "h1": tkfont.Font(family="Segoe UI", size=20, weight="bold"),
            "h2": tkfont.Font(family="Segoe UI", size=14, weight="bold"),
            "body": tkfont.Font(family="Segoe UI", size=11),
            "code": tkfont.Font(family="Consolas", size=10),
            "bold": tkfont.Font(family="Segoe UI", size=11, weight="bold")
        }

    def _create_content(self):
        # --- Scrollable Canvas Setup ---
        canvas = tk.Canvas(self, bg=self.colors["bg_main"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        
        # The main frame inside the canvas
        self.main_frame = tk.Frame(canvas, bg=self.colors["bg_main"])

        self.main_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        
        # Responsive width handling
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- CONTENT GENERATION ---
        
        # Header
        self._create_header_banner("🧠 PINNs: A Visual Guide")

        # Card 1: What is a Neural Network?
        card1 = self._create_card(self.main_frame)
        self._add_card_title(card1, "Lesson 1: The 'Black Box' (Neural Networks)")
        self._add_text(card1, 
            "Imagine a Neural Network (NN) as a machine with thousands of tuning knobs (parameters). "
            "You feed a number in one side (e.g., time t=0) and it spits out a prediction "
            "(e.g., position x=1.2).")
        
        self._add_diagram(card1, "[ Input: t ]  ──>  [ 🕸️ NEURAL NET ]  ──>  [ Output: x ]")
        
        self._add_text(card1,
            "Initially, the machine is 'dumb'. It guesses randomly. To teach it, we need three things:")
        
        self._add_bullet(card1, "Layers & Neurons", "The brain structure. More layers = deeper understanding of complex patterns.")
        self._add_bullet(card1, "Activation Functions", "Mathematical sparks that allow the network to learn curves, not just straight lines.")
        self._add_bullet(card1, "Backpropagation", "The learning process. It compares the guess to the real answer, calculates the error, and adjusts the knobs to do better next time.")

        # Card 2: The Problem
        card2 = self._create_card(self.main_frame)
        self._add_card_title(card2, "Lesson 2: Why standard AI fails in Physics")
        self._add_text(card2,
            "Standard AI is 'Data Hungry'. It only knows what you show it. If you train it on a pendulum swinging "
            "for 10 seconds, it has no idea what happens at second 11. It might predict the pendulum flies "
            "into space because it doesn't know gravity exists.")

        # Card 3: The PINN Solution
        card3 = self._create_card(self.main_frame)
        self._add_card_title(card3, "Lesson 3: The Magic of PINNs")
        self._add_text(card3,
            "A Physics-Informed Neural Network (PINN) is different. We don't just punish the network "
            "for missing the data points. We also punish it if it violates the Laws of Physics.")
        
        self._add_code_block(card3, "TOTAL LOSS = DATA ERROR + PHYSICS ERROR")
        
        self._add_text(card3, 
            "• Data Error: How far is the prediction from the experimental measurements?\n"
            "• Physics Error: We plug the prediction into the Differential Equation (e.g., F=ma). If the equation doesn't balance to zero, the network gets a penalty.")
        
        self._add_highlight_box(card3, 
            "💡 The Superpower: Because the network knows the laws of physics, it can solve problems "
            "with very little data, or sometimes NO data at all!")

        # Card 4: Hyperparameters Guide
        card4 = self._create_card(self.main_frame)
        self._add_card_title(card4, "🎛️ Configuration Guide (The Buttons)")
        
        terms = [
            ("Epochs", "How many times the network loops through the entire learning process. 15,000 epochs means it tried to improve 15,000 times."),
            ("Learning Rate", "The step size for corrections. High (0.1) = wild jumps. Low (0.0001) = tiny steps. We usually want a balance (e.g., 0.001)."),
            ("Hidden Layers", "The depth of the brain. Simple spring problems need 3-4 layers. Complex heat flows need 6+."),
            ("Activation (Tanh)", "For Physics, 'Tanh' is king. Unlike 'ReLU' (used in image AI), Tanh is smooth and curvy, perfect for derivatives.")
        ]
        
        for term, desc in terms:
            self._add_definition(card4, term, desc)

        # Footer padding
        tk.Frame(self.main_frame, height=30, bg=self.colors["bg_main"]).pack()

    # --- UI HELPER METHODS (The Styling Engine) ---

    def _create_header_banner(self, text):
        """Creates a full-width header."""
        frame = tk.Frame(self.main_frame, bg=self.colors["accent"], height=80)
        frame.pack(fill="x", pady=(0, 20))
        frame.pack_propagate(False) # Force height
        
        lbl = tk.Label(frame, text=text, font=("Segoe UI", 24, "bold"), 
                      bg=self.colors["accent"], fg="white")
        lbl.pack(side="left", padx=30, pady=20)

    def _create_card(self, parent):
        """Creates a white card with a subtle visual lift."""
        # Outer container for padding
        outer = tk.Frame(parent, bg=self.colors["bg_main"], padx=20, pady=10)
        outer.pack(fill="x")
        
        # The Card itself
        card = tk.Frame(outer, bg=self.colors["card_bg"], bd=1, relief="flat")
        card.pack(fill="x", ipadx=20, ipady=15)
        
        # A subtle bottom border to simulate shadow/depth
        tk.Frame(outer, bg="#D1D5DB", height=2).pack(fill="x", padx=2) # Shadow line
        
        return card

    def _add_card_title(self, parent, text):
        tk.Label(parent, text=text, font=self.fonts["h2"], 
                bg=self.colors["card_bg"], fg=self.colors["text_header"]).pack(anchor="w", pady=(0, 15))
        
        # Separator line
        tk.Frame(parent, bg=self.colors["accent"], height=2, width=50).pack(anchor="w", pady=(0, 15))

    def _add_text(self, parent, text):
        tk.Label(parent, text=text, font=self.fonts["body"], 
                bg=self.colors["card_bg"], fg=self.colors["text_body"], 
                justify="left", wraplength=900).pack(anchor="w", pady=(0, 10))

    def _add_diagram(self, parent, text):
        """ASCII Diagram styling"""
        frame = tk.Frame(parent, bg=self.colors["bg_main"], bd=0)
        frame.pack(fill="x", pady=10)
        
        tk.Label(frame, text=text, font=self.fonts["code"], 
                bg=self.colors["bg_main"], fg="#555").pack(pady=10)

    def _add_bullet(self, parent, title, desc):
        frame = tk.Frame(parent, bg=self.colors["card_bg"])
        frame.pack(fill="x", pady=4)
        
        # Bullet dot
        tk.Label(frame, text="•", font=("Arial", 16), fg=self.colors["accent"], 
                bg=self.colors["card_bg"]).pack(side="left", anchor="n")
        
        # Content
        content_frame = tk.Frame(frame, bg=self.colors["card_bg"])
        content_frame.pack(side="left", fill="x", padx=10)
        
        tk.Label(content_frame, text=title, font=self.fonts["bold"], 
                bg=self.colors["card_bg"], fg=self.colors["text_header"]).pack(anchor="w")
        tk.Label(content_frame, text=desc, font=self.fonts["body"], 
                bg=self.colors["card_bg"], fg=self.colors["text_body"], 
                wraplength=850, justify="left").pack(anchor="w")

    def _add_code_block(self, parent, text):
        frame = tk.Frame(parent, bg=self.colors["code_bg"])
        frame.pack(fill="x", pady=10, padx=10)
        
        tk.Label(frame, text=text, font=("Consolas", 12, "bold"), 
                bg=self.colors["code_bg"], fg=self.colors["code_fg"]).pack(pady=10, padx=10)

    def _add_highlight_box(self, parent, text):
        frame = tk.Frame(parent, bg=self.colors["highlight"], 
                        bd=1, relief="solid", highlightbackground=self.colors["accent"])
        frame.pack(fill="x", pady=15, padx=5)
        
        tk.Label(frame, text=text, font=self.fonts["body"], 
                bg=self.colors["highlight"], fg="#004d40", 
                justify="left", wraplength=850).pack(pady=10, padx=10)

    def _add_definition(self, parent, term, desc):
        frame = tk.Frame(parent, bg=self.colors["card_bg"])
        frame.pack(fill="x", pady=8)
        
        tk.Label(frame, text=term, font=self.fonts["bold"], 
                bg=self.colors["card_bg"], fg=self.colors["accent"]).pack(anchor="w")
        tk.Label(frame, text=desc, font=self.fonts["body"], 
                bg=self.colors["card_bg"], fg=self.colors["text_body"], 
                wraplength=900, justify="left").pack(anchor="w", padx=10)

    def on_shared_state_change(self, key, value):
        pass