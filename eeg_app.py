import tkinter as tk
from tkinter import ttk

class EEGApp:
    def __init__(self, master, callback_ini, callback_mode):
        """Initialize the EEG application with the main window."""
        self.master = master
        self.callback = callback_ini  # Store the callback function
        self.callback_mode = callback_mode
        self.master.title("EEG Data Input")

        # Initialize variables
        self.channels = ["F3", "F4", "C3", "Cz", "C4", "P3", "P4"]

        self.channel_var = tk.StringVar(value="F3")
        self.lower_cutoff_var = tk.DoubleVar(value=5)
        self.upper_cutoff_var = tk.DoubleVar(value=50)

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        """Create widgets for the GUI."""
        # Channel selection
        channel_label = ttk.Label(self.master, text="Select Channel:")
        channel_label.grid(column=0, row=0, padx=10, pady=10)
        channel_combo = ttk.Combobox(self.master, textvariable=self.channel_var, values=["F3", "F4", "C3", "Cz", "C4", "P3", "P4"])
        channel_combo.grid(column=1, row=0, padx=10, pady=10)

        # Lower cutoff frequency
        lower_cutoff_label = ttk.Label(self.master, text="Lower Cutoff Hz:")
        lower_cutoff_label.grid(column=0, row=1, padx=10, pady=10)
        lower_cutoff_entry = ttk.Entry(self.master, textvariable=self.lower_cutoff_var)
        lower_cutoff_entry.grid(column=1, row=1, padx=10, pady=10)

        # Upper cutoff frequency
        upper_cutoff_label = ttk.Label(self.master, text="Upper Cutoff Hz:")
        upper_cutoff_label.grid(column=0, row=2, padx=10, pady=10)
        upper_cutoff_entry = ttk.Entry(self.master, textvariable=self.upper_cutoff_var)
        upper_cutoff_entry.grid(column=1, row=2, padx=10, pady=10)

        # Submit button
        submit_button = ttk.Button(self.master, text="Submit", command=self.submit)
        submit_button.grid(column=0, row=3, columnspan=2, padx=10, pady=10)

    def move_window_to_right(self):
        # Calculate new x position to move the window to the right side of the screen
        screen_width = self.master.winfo_screenwidth()
        new_x_position = screen_width - 600
        self.master.geometry(f"+{new_x_position}+50")
    def submit(self):
        """Handle the submit button click event."""
        # Clear the initial window
        for widget in self.master.winfo_children():
            widget.destroy()

        # Create the new window layout
        self.master.title("Response Page")
        self.labels = []
        for i in range(3):
            label = ttk.Label(self.master, text=str(i + 1), relief="solid", width=10)
            label.grid(column=i, row=0, padx=5, pady=20)
            self.labels.append(label)
        for i in range(3):
            self.master.grid_columnconfigure(i, weight=1)

        # Center labels on the window
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=1)

        # Bind key events to light up labels
        self.master.bind("1", lambda e: self.activate_label(0))
        self.master.bind("2", lambda e: self.activate_label(1))
        self.master.bind("3", lambda e: self.activate_label(2))
        self.activate_label(0)

        # Directional buttons
        self.create_directional_buttons()

        self.move_window_to_right()
        # Retrieve the input values
        selected_channel = self.channel_var.get()
        selected_channel_index = self.channels.index(selected_channel)
        lower_cutoff = self.lower_cutoff_var.get()
        upper_cutoff = self.upper_cutoff_var.get()
        print(f"Selected Channel: {selected_channel}, Lower Cutoff: {lower_cutoff} Hz, Upper Cutoff: {upper_cutoff} Hz")
        # Call the callback function with the data
        self.callback(selected_channel_index, lower_cutoff, upper_cutoff)

    def create_directional_buttons(self):
        button_frame = ttk.Frame(self.master)
        # Make sure the button_frame is centered by setting columnspan to the number of columns used by the labels
        button_frame.grid(column=0, row=1, columnspan=3, pady=20)

        # Buttons
        self.up_button = ttk.Button(button_frame, text="Up")
        self.up_button.grid(column=1, row=0)

        self.left_button = ttk.Button(button_frame, text="Left")
        self.left_button.grid(column=0, row=1)

        self.space_button = ttk.Button(button_frame, text="Space")
        self.space_button.grid(column=1, row=1)

        self.right_button = ttk.Button(button_frame, text="Right")
        self.right_button.grid(column=2, row=1)

        self.down_button = ttk.Button(button_frame, text="Down")
        self.down_button.grid(column=1, row=2)

        # Configure the button frame's grid to distribute space evenly
        for i in range(3):
            button_frame.grid_columnconfigure(i, weight=1)
            button_frame.grid_rowconfigure(i, weight=1)

        # Bind keys
        #self.master.bind("<Up>", lambda e: self.highlight_button(self.up_button))
        #self.master.bind("<Down>", lambda e: self.highlight_button(self.down_button))
        #self.master.bind("<Left>", lambda e: self.highlight_button(self.left_button))
        #self.master.bind("<Right>", lambda e: self.highlight_button(self.right_button))
        #self.master.bind("<space>", lambda e: self.highlight_button(self.space_button))

    def highlight_button(self, button):
        """Highlight a button when a key is pressed."""
        button.state(['pressed'])
        self.master.after(100, lambda: button.state(['!pressed']))
    def activate_label(self, index):
        """Activate the label corresponding to the pressed key."""
        for label in self.labels:
            label.config(background="SystemButtonFace")  # Reset background
        self.labels[index].config(background="yellow")  # Highlight the selected label
        self.callback_mode(index + 1)
