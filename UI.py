import tkinter as tk
from tkinter import ttk
import webbrowser
from PIL import Image, ImageTk

corrected_label = None  # Global variable to track the corrected query label

def toggle_preview(event, preview_label, toggle_label):
    if preview_label.winfo_ismapped():
        preview_label.pack_forget()
        toggle_label.config(text="More")
    else:
        preview_label.pack(fill=tk.X, pady=5)
        toggle_label.config(text="Less")

def search(event=None):
    query = entry.get()
    if query:
        if corrected_label:
            corrected_label.destroy()
        show_loading_indicator()
        process_query_callback(query, use_spellcheck=True)

def search_without_spellcheck(query):
    global corrected_label
    if query:
        show_loading_indicator()
        process_query_callback(query, use_spellcheck=False)
    if corrected_label:
        corrected_label.destroy()
        corrected_label = None

def show_loading_indicator():
    for widget in results_frame.winfo_children():
        widget.destroy()
    loading_label = tk.Label(results_frame, text="Loading...", bg="white", font=("Helvetica", 12, "italic"))
    loading_label.pack(pady=20)

def update_results(new_query, topics, results):
    global entry, corrected_label, logo_label, description_label  # Ensure 'entry' and 'corrected_label' are correctly referenced
    entry.delete(0, tk.END)
    entry.insert(0, new_query)
    
    # Clear previous results
    for widget in results_frame.winfo_children():
        widget.destroy()

    if corrected_label:
        corrected_label.destroy()
        corrected_label = None

    if not results:
        no_results_frame = tk.Frame(results_frame, bg="white", pady=20)
        no_results_frame.pack(fill=tk.BOTH, expand=True)

        no_results_label = tk.Label(no_results_frame, text="No results found", bg="white", font=("Helvetica", 14, "bold"))
        no_results_label.pack(pady=10)

        no_results_icon = tk.Label(no_results_frame, text="😞", bg="white", font=("Helvetica", 50))
        no_results_icon.pack()
    else:
        display_results(results)

def display_results(results):
    for result in results:
        result_frame = tk.Frame(results_frame, bg="white", bd=0, relief="solid", padx=10, pady=5)
        result_frame.pack(fill=tk.X, pady=5)

        title_label = tk.Label(result_frame, text=f"Name of site: {result['Name of site']}", bg="white", font=("Helvetica", 14, "bold"))
        title_label.pack(anchor="w")

        url_label = tk.Label(result_frame, text=f"Url: {result['Url']}", bg="white", font=("Helvetica", 12), fg="blue", cursor="hand2")
        url_label.pack(anchor="w")
        url_label.bind("<Button-1>", lambda e, url=result['Url']: webbrowser.open(url))

        keywords_label = tk.Label(result_frame, text=f"Keywords: {result['Keywords']}", bg="white", font=("Helvetica", 12))
        keywords_label.pack(anchor="w")

        toggle_label = tk.Label(result_frame, text="More", bg="white", font=("Helvetica", 12), fg="blue", cursor="hand2")
        toggle_label.pack(anchor="w", pady=(5, 0))

        preview_label = tk.Label(result_frame, text=result['Preview'], bg="#f9f9f9", font=("Helvetica", 12), wraplength=480, padx=10, pady=5, relief="solid", bd=1)
        preview_label.pack_forget()
        
        result_frame.bind("<Button-1>", lambda e, pl=preview_label, tl=toggle_label: toggle_preview(e, pl, tl))
        toggle_label.bind("<Button-1>", lambda e, pl=preview_label, tl=toggle_label: toggle_preview(e, pl, tl))

def display_corrected_query(old_query, new_query):
    global corrected_label
    entry.delete(0, tk.END)
    entry.insert(0, new_query)
    
    if corrected_label:
        corrected_label.destroy()
    
    corrected_label = tk.Label(search_frame, text=f"Or did you really want to search for '{old_query}'?", fg="blue", cursor="hand2", font=("Helvetica", 12, "italic"))
    corrected_label.pack(side=tk.LEFT, anchor='w', pady=5)
    corrected_label.bind("<Button-1>", lambda e: search_without_spellcheck(old_query))

def start_ui(process_query_cb):
    global root, entry, results_frame, process_query_callback, search_frame, corrected_label, logo_label, description_label  # Ensure 'entry' and 'process_query_callback' are defined globally here
    process_query_callback = process_query_cb

    # Create the main window
    root = tk.Tk()
    root.title("Modern Search Engine")
    root.state('zoomed')  # Open in full-screen mode

    # Set the background color
    root.configure(bg="#f0f0f0")

    # Create a style for the widgets
    style = ttk.Style()
    style.configure("TEntry", padding=6, relief="flat", background="white", font=("Helvetica", 14))
    style.configure("TLabel", background="white", font=("Helvetica", 12))
    style.configure("TFrame", background="white")

    # Add a logo
    logo = Image.open("path/to/your/logo.png")  # Replace with the path to your logo
    logo = logo.resize((150, 150), Image.ANTIALIAS)
    logo_image = ImageTk.PhotoImage(logo)
    logo_label = tk.Label(root, image=logo_image, bg="#f0f0f0")
    logo_label.image = logo_image  # Keep a reference to avoid garbage collection
    logo_label.pack(pady=20)

    # Add a description label
    description_label = tk.Label(root, text="Welcome to the Modern Search Engine. Enter your query below and press Enter to search.", bg="#f0f0f0", font=("Helvetica", 14))
    description_label.pack(pady=10)

    # Create a central frame for the search bar
    search_frame = tk.Frame(root, bg="#f0f0f0")
    search_frame.pack(pady=20)

    entry = tk.Entry(search_frame, width=50, font=("Helvetica", 14), bd=0, relief="flat", highlightthickness=1, highlightbackground="#d9d9d9")
    entry.pack(side=tk.TOP, padx=10, ipady=6)
    entry.insert(0, "Enter your search query here...")

    def on_entry_click(event):
        if entry.get() == "Enter your search query here...":
            entry.delete(0, "end")  # delete all the text in the entry
            entry.insert(0, "")  # Insert blank for user input

    def on_focusout(event):
        if entry.get() == "":
            entry.insert(0, "Enter your search query here...")

    entry.bind("<FocusIn>", on_entry_click)
    entry.bind("<FocusOut>", on_focusout)

    # Bind the Enter key to the search function
    entry.bind('<Return>', search)

    # Create a frame to display search results with a scrollbar
    results_container = tk.Frame(root, bg="#f0f0f0")
    results_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    canvas = tk.Canvas(results_container, bg="#f0f0f0")
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas.configure(yscrollcommand=scrollbar.set)

    results_frame = tk.Frame(canvas, bg="#f0f0f0")
    canvas_window = canvas.create_window((0, 0), window=results_frame, anchor="nw")

    def on_canvas_configure(event):
        canvas.itemconfig(canvas_window, width=event.width - scrollbar.winfo_width())  # Adjust canvas width

    canvas.bind('<Configure>', on_canvas_configure)

    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    results_frame.bind('<Configure>', on_frame_configure)

    # Bind mouse wheel events to the canvas
    def on_mouse_wheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", on_mouse_wheel)
    canvas.bind_all("<Button-4>", on_mouse_wheel)
    canvas.bind_all("<Button-5>", on_mouse_wheel)

    # Run the application
    root.mainloop()

if __name__ == "__main__":
    start_ui(fetch_and_display_results)  # Using the actual function for processing queries
