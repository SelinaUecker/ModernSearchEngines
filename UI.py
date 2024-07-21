import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
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
    loading_label = ttk.Label(results_frame, text="Loading...", style="TLabel")
    loading_label.pack(pady=20)

def update_results(new_query, topics, results):
    global entry, corrected_label  # Ensure 'entry' and 'corrected_label' are correctly referenced
    entry.delete(0, tk.END)
    entry.insert(0, new_query)
    
    # Clear previous results
    for widget in results_frame.winfo_children():
        widget.destroy()

    if not results:
        no_results_frame = ttk.Frame(results_frame, style="TFrame")
        no_results_frame.pack(fill=tk.BOTH, expand=True, pady=20)

        no_results_label = ttk.Label(no_results_frame, text="No results found", style="TLabel")
        no_results_label.pack(pady=10)

        no_results_icon = ttk.Label(no_results_frame, text="😞", style="TLabel")
        no_results_icon.pack()
    else:
        display_results(results)

    # Scroll to the top of the results
    canvas.yview_moveto(0)

def display_results(results):
    for result in results:
        result_frame = ttk.Frame(results_frame, style="Transparent.TFrame", padding=(10, 5))
        result_frame.pack(fill=tk.X, pady=5)

        title_label = ttk.Label(result_frame, text=result['Name of site'], style="Transparent.TLabel", font=("Helvetica", 14, "bold"))
        title_label.pack(anchor="w")

        url_label = ttk.Label(result_frame, text=result['Url'], style="Transparent.TLabel", font=("Helvetica", 12), foreground="blue", cursor="hand2")
        url_label.pack(anchor="w")
        url_label.bind("<Button-1>", lambda e, url=result['Url']: [webbrowser.open(url), canvas.yview_moveto(0)])

        keywords_label = ttk.Label(result_frame, text=result['Keywords'], style="Transparent.TLabel", font=("Helvetica", 12))
        keywords_label.pack(anchor="w")

        toggle_label = ttk.Label(result_frame, text="More", style="Transparent.TLabel", font=("Helvetica", 12), foreground="blue", cursor="hand2")
        toggle_label.pack(anchor="w", pady=(5, 0))

        preview_label = ttk.Label(result_frame, text=result['Preview'], style="Transparent.TLabel", wraplength=480, padding=(10, 5), relief="solid", borderwidth=1)
        preview_label.pack_forget()
        
        result_frame.bind("<Button-1>", lambda e, pl=preview_label, tl=toggle_label: toggle_preview(e, pl, tl))
        toggle_label.bind("<Button-1>", lambda e, pl=preview_label, tl=toggle_label: toggle_preview(e, pl, tl))

def display_corrected_query(old_query, new_query):
    global corrected_label
    entry.delete(0, tk.END)
    entry.insert(0, new_query)
    
    if corrected_label:
        corrected_label.destroy()
    
    corrected_label = ttk.Label(corrected_label_container, text=f"Or did you really want to search for '{old_query}'?", style="TLabel", foreground="blue", cursor="hand2")
    corrected_label.pack(pady=5)
    corrected_label.bind("<Button-1>", lambda e: search_without_spellcheck(old_query))

def start_ui(process_query_cb):
    global root, entry, results_frame, process_query_callback, search_frame, corrected_label, canvas, background_label, background_photo, background_image, corrected_label_container  # Ensure 'entry' and 'process_query_callback' are defined globally here
    process_query_callback = process_query_cb

    # Create the main window with a theme
    root = ThemedTk(theme='Adapta')  # You can change the theme here
    root.title("Modern Search Engine")
    root.state('zoomed')  # Open in full-screen mode

    # Load background image
    background_image = Image.open("background.jpg")

    # Create a label to display the background image
    background_label = tk.Label(root)
    background_label.place(relwidth=1, relheight=1)

    def resize_background(event):
        global background_photo
        new_width = event.width
        new_height = int(new_width * background_image.height / background_image.width)
        resized_image = background_image.resize((new_width, new_height), Image.ANTIALIAS)
        background_photo = ImageTk.PhotoImage(resized_image)
        background_label.config(image=background_photo)

    root.bind('<Configure>', resize_background)

    # Set the background color
    root.configure(bg="#f0f0f0")

    # Create a style for the widgets
    style = ttk.Style()
    style.configure("TEntry", padding=6, relief="flat", font=("Helvetica", 14), borderwidth=0)
    style.configure("TLabel", background="white", font=("Helvetica", 12))
    style.configure("TFrame", background="white")
    style.configure("Search.TEntry", padding=6, relief="flat", font=("Helvetica", 14), borderwidth=0)

    # Configure transparent style
    style.configure("Transparent.TFrame", background="white")
    style.configure("Transparent.TLabel", background="white")

    # Create a central frame for the search bar
    search_frame = ttk.Frame(root, style="Transparent.TFrame")
    search_frame.pack(pady=50, padx=20)

    # Add the search engine name to the left of the search bar
    search_label = ttk.Label(search_frame, text="TüSearch", font=("Helvetica", 20, "bold"), style="Transparent.TLabel")
    search_label.grid(row=0, column=0, padx=10, pady=6, sticky="w")

    entry = ttk.Entry(search_frame, width=50, style="Search.TEntry")
    entry.grid(row=0, column=1, padx=10, pady=6)
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

    # Create a fixed-height container for the corrected label to prevent content shifting
    corrected_label_container = ttk.Frame(root, style="Transparent.TFrame", height=30)
    corrected_label_container.pack(fill=tk.X, padx=10)

    # Create a frame to display search results with a scrollbar
    results_container = ttk.Frame(root, style="Transparent.TFrame")
    results_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    canvas = tk.Canvas(results_container, bg="#f0f0f0")
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas.configure(yscrollcommand=scrollbar.set)

    results_frame = ttk.Frame(canvas, style="Transparent.TFrame")
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
