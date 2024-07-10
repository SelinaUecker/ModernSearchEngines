import tkinter as tk
from tkinter import ttk
from transformers import pipeline
import re
import webbrowser

# Load the pipeline for spell correction
fix_spelling = pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base")

def normalize_text(text):
    # Remove punctuation and convert to lowercase
    return re.sub(r'[^\w\s]', '', text).lower()

def correct_text(text):
    result = fix_spelling(text, max_length=2048)
    corrected_text = result[0]['generated_text']
    # Remove trailing period
    corrected_text = corrected_text.rstrip('.')
    return corrected_text

def toggle_preview(event, preview_label, toggle_label):
    if preview_label.winfo_ismapped():
        preview_label.pack_forget()
        toggle_label.config(text="More")
    else:
        preview_label.pack(fill=tk.X, pady=5)
        toggle_label.config(text="Less")

def search(event=None, query=None):
    if query is None:
        query = entry.get()

    if not query:
        return  # Do nothing if the input field is empty

    # Clear previous spell suggestion
    spell_suggestion_label.config(text="")
    spell_suggestion_label.unbind("<Button-1>")

    # Correct the query
    corrected_query = correct_text(query)

    # Normalize both the original and corrected queries to check for meaningful changes
    normalized_original = normalize_text(query)
    normalized_corrected = normalize_text(corrected_query)

    if normalized_corrected != normalized_original:
        spell_suggestion_label.config(
            text=f"Did you mean '{corrected_query}'? Click here to search."
        )
        spell_suggestion_label.bind("<Button-1>", lambda e: update_query(corrected_query))

    # Simulate search results
    results = [
        {
            "Name of site": f"Site {i+1}",
            "Url": f"https://www.site{i+1}.com",
            "Keywords": f"keyword1, keyword2, keyword{i+1}",
            "Preview": "This is a dummy preview text for the website content. It provides a brief summary of what the website is about."
        } for i in range(10)
    ]
    
    # Clear previous results
    for widget in results_frame.winfo_children():
        widget.destroy()
    
    # Display new results
    for result in results:
        result_frame = tk.Frame(results_frame, bg="white", bd=2, relief="solid", padx=10, pady=5)
        result_frame.pack(fill=tk.X, pady=5)

        title_label = tk.Label(result_frame, text=f"Name of site: {result['Name of site']}", bg="white", font=("Arial", 12, "bold"))
        title_label.pack(anchor="w")

        url_label = tk.Label(result_frame, text=f"Url: {result['Url']}", bg="white", font=("Arial", 10), fg="blue", cursor="hand2")
        url_label.pack(anchor="w")
        url_label.bind("<Button-1>", lambda e, url=result['Url']: webbrowser.open(url))

        keywords_label = tk.Label(result_frame, text=f"Keywords: {result['Keywords']}", bg="white", font=("Arial", 10))
        keywords_label.pack(anchor="w")

        toggle_label = tk.Label(result_frame, text="More", bg="white", font=("Arial", 10), fg="blue", cursor="hand2")
        toggle_label.pack(anchor="w", pady=(5, 0))

        preview_label = tk.Label(result_frame, text=result['Preview'], bg="white", font=("Arial", 10), wraplength=480, padx=10, pady=5, relief="solid", bd=1)
        preview_label.pack_forget()
        
        result_frame.bind("<Button-1>", lambda e, pl=preview_label, tl=toggle_label: toggle_preview(e, pl, tl))
        toggle_label.bind("<Button-1>", lambda e, pl=preview_label, tl=toggle_label: toggle_preview(e, pl, tl))

def update_query(corrected_query):
    entry.delete(0, tk.END)
    entry.insert(0, corrected_query)
    search(query=corrected_query)

# Create the main window
root = tk.Tk()
root.title("Simple Search Engine")
root.geometry("600x600")  # Fixed window size
root.resizable(False, False)

# Set the background color
root.configure(bg="white")

# Create a style for the widgets
style = ttk.Style()
style.configure("TEntry", padding=6, relief="flat", background="white")
style.configure("TButton", padding=6, relief="flat", background="light blue", foreground="white", font=("Arial", 10, "bold"))
style.configure("TLabel", background="white", font=("Arial", 10))
style.configure("TListbox", background="white", font=("Arial", 10))

# Create an entry widget for the search query
entry = ttk.Entry(root, width=50)
entry.pack(pady=10)

# Bind the Enter key to the search function
entry.bind('<Return>', search)

# Create a label for spell check suggestion
spell_suggestion_label = ttk.Label(root, foreground="blue", cursor="hand2")
spell_suggestion_label.pack(pady=5)

# Create a frame to display search results with a scrollbar
results_container = tk.Frame(root, bg="light blue")
results_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

canvas = tk.Canvas(results_container, bg="light blue")
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)

results_frame = tk.Frame(canvas, bg="light blue")
canvas_window = canvas.create_window((0, 0), window=results_frame, anchor="nw")

def on_canvas_configure(event):
    canvas.itemconfig(canvas_window, width=event.width)

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
