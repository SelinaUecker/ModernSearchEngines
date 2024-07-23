import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
import webbrowser
from PIL import Image, ImageTk

corrected_label = None 

def toggle_preview(event, preview_label, toggle_label):
    """
    Toggle the visibility of the preview label.

    This function is used to manage the display of additional information
    about each search result. When a user clicks on a search result, this function
    toggles between showing and hiding the preview text associated with that result.
    The toggle_label changes between "More" and "Less" to indicate the current state.

    Args:
        event (tk.Event): The event that triggered the function.
        preview_label (ttk.Label): The label containing the preview text.
        toggle_label (ttk.Label): The label that displays "More" or "Less".

    Returns:
        None
    """
    if preview_label.winfo_ismapped():
        preview_label.pack_forget()
        toggle_label.config(text="More")
    else:
        preview_label.pack(fill=tk.X, pady=5)
        toggle_label.config(text="Less")

def search(event=None):
    """
    Trigger a search with spellcheck enabled.

    This function is called when the user submits a search query by pressing
    the Enter key. It initiates a search operation, using
    a spellcheck feature to correct potential typos in the query. The function also
    shows a loading indicator while the search results are being fetched.

    Args:
        event (tk.Event, optional): The event that triggered the function. Defaults to None.

    Returns:
        None
    """
    query = entry.get()
    if query:
        if corrected_label:
            corrected_label.destroy()
        show_loading_indicator()
        process_query_callback(query, use_spellcheck=True)

def search_without_spellcheck(query):
    """
    Trigger a search without spellcheck.

    This function performs a search using the original query provided by the user,
    bypassing any spellcheck corrections. It is particularly useful when the user
    wants to search for a specific term that might be incorrectly altered by spellcheck.

    Args:
        query (str): The search query.

    Returns:
        None
    """
    global corrected_label
    if query:
        show_loading_indicator()
        process_query_callback(query, use_spellcheck=False)
    if corrected_label:
        corrected_label.destroy()
        corrected_label = None

def show_loading_indicator():
    """
    Show a loading indicator while fetching results.

    This function clears any existing search results and displays a "Loading..."
    message to inform the user that their search query is being processed. This
    provides visual feedback and improves the user experience by indicating that
    the application is working on their request.

    Args:
        None

    Returns:
        None
    """
    for widget in results_frame.winfo_children():
        widget.destroy()
    loading_label = ttk.Label(results_frame, text="Loading...", style="TLabel")
    loading_label.pack(pady=20)

def update_results(new_query, topics, results):
    """
    Update the results displayed based on the search query.

    This function is called to refresh the UI with the latest search results.
    It updates the search entry with the corrected query, clears any previous
    results, and displays new search results. If no results are found, it shows
    a "No results found" message.

    Args:
        new_query (str): The corrected search query.
        topics (list): List of related topics (not used in this version).
        results (list): List of search results.

    Returns:
        None
    """
    global entry, corrected_label  
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

    canvas.yview_moveto(0)

def display_results(results):
    """
    Display the search results.

    This function renders each search result in the UI. It creates a frame for each result,
    displaying the site name, URL, keywords, and a preview of the content. Users can click
    on a result to toggle additional preview information. Each URL is clickable, opening
    the link in a web browser.

    Args:
        results (list): List of dictionaries containing search result data.

    Returns:
        None
    """
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
    """
    Display the corrected query suggestion.

    This function suggests a corrected query to the user when the initial query
    appears to contain typos or errors. It updates the search entry with the corrected
    query and provides a clickable suggestion to revert to the original query if desired.

    Args:
        old_query (str): The original search query.
        new_query (str): The corrected search query.

    Returns:
        None
    """
    global corrected_label
    entry.delete(0, tk.END)
    entry.insert(0, new_query)
    
    if corrected_label:
        corrected_label.destroy()
    
    corrected_label = ttk.Label(corrected_label_container, text=f"Or did you really want to search for '{old_query}'?", style="TLabel", foreground="blue", cursor="hand2")
    corrected_label.pack(pady=5)
    corrected_label.bind("<Button-1>", lambda e: search_without_spellcheck(old_query))

def start_ui(process_query_cb):
    """
    Initialize the UI and start the main loop.

    This function sets up the main user interface for the search application.
    It includes the search bar, results display area, and other UI elements.
    The function also starts the main application loop, keeping the interface responsive
    and interactive.

    Args:
        process_query_cb (function): The callback function to process search queries.

    Returns:
        None
    """
    global root, entry, results_frame, process_query_callback, search_frame, corrected_label, canvas, background_label, background_photo, background_image, corrected_label_container
    process_query_callback = process_query_cb

    root = ThemedTk(theme='Adapta')  
    root.title("Modern Search Engine")
    root.state('zoomed')  

    background_image = Image.open("background.jpg")

    background_label = tk.Label(root)
    background_label.place(relwidth=1, relheight=1)

    def resize_background(event):
        """
        Resize the background image to fit the window.

        This function ensures that the background image dynamically resizes to fit the
        application window, maintaining a visually appealing layout. It is called whenever
        the window is resized.

        Args:
            event (tk.Event): The event that triggered the function.

        Returns:
            None
        """
        global background_photo
        new_width = event.width
        new_height = int(new_width * background_image.height / background_image.width)
        resized_image = background_image.resize((new_width, new_height), Image.LANCZOS)
        background_photo = ImageTk.PhotoImage(resized_image)
        background_label.config(image=background_photo)

    root.bind('<Configure>', resize_background)

    root.configure(bg="#f0f0f0")

    style = ttk.Style()
    style.configure("TEntry", padding=6, relief="flat", font=("Helvetica", 14), borderwidth=0)
    style.configure("TLabel", background="white", font=("Helvetica", 12))
    style.configure("TFrame", background="white")
    style.configure("Search.TEntry", padding=6, relief="flat", font=("Helvetica", 14), borderwidth=0)

    style.configure("Transparent.TFrame", background="white")
    style.configure("Transparent.TLabel", background="white")

    search_frame = ttk.Frame(root, style="Transparent.TFrame")
    search_frame.pack(pady=50, padx=20)

    search_label = ttk.Label(search_frame, text="TüSearch", font=("Helvetica", 20, "bold"), style="Transparent.TLabel")
    search_label.grid(row=0, column=0, padx=10, pady=6, sticky="w")

    entry = ttk.Entry(search_frame, width=50, style="Search.TEntry")
    entry.grid(row=0, column=1, padx=10, pady=6)
    entry.insert(0, "Enter your search query here...")

    def on_entry_click(event):
        """
        Clear placeholder text on entry click.

        This function ensures that the placeholder text in the search entry is cleared
        when the user clicks on it, allowing them to start typing their query without
        having to manually delete the placeholder text.

        Args:
            event (tk.Event): The event that triggered the function.

        Returns:
            None
        """
        if entry.get() == "Enter your search query here...":
            entry.delete(0, "end") 
            entry.insert(0, "")  

    def on_focusout(event):
        """
        Restore placeholder text if entry is empty.

        This function restores the placeholder text in the search entry if it is left empty
        when the user clicks away, maintaining a clear prompt for the user to enter their search query.

        Args:
            event (tk.Event): The event that triggered the function.

        Returns:
            None
        """
        if entry.get() == "":
            entry.insert(0, "Enter your search query here...")

    entry.bind("<FocusIn>", on_entry_click)
    entry.bind("<FocusOut>", on_focusout)

    entry.bind('<Return>', search)

    corrected_label_container = ttk.Frame(root, style="Transparent.TFrame", height=30)
    corrected_label_container.pack(fill=tk.X, padx=10)

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
        """
        Adjust canvas width to fit the window.

        This function ensures that the canvas width is adjusted to fit the window size,
        maintaining a consistent and responsive layout. It is called whenever the canvas
        is resized.

        Args:
            event (tk.Event): The event that triggered the function.

        Returns:
            None
        """
        canvas.itemconfig(canvas_window, width=event.width - scrollbar.winfo_width())

    canvas.bind('<Configure>', on_canvas_configure)

    def on_frame_configure(event):
        """
        Update the scroll region.

        This function updates the scroll region of the canvas, ensuring that all content
        can be scrolled through properly. It is called whenever the contents of the results
        frame are resized.

        Args:
            event (tk.Event): The event that triggered the function.

        Returns:
            None
        """
        canvas.configure(scrollregion=canvas.bbox("all"))

    results_frame.bind('<Configure>', on_frame_configure)

    def on_mouse_wheel(event):
        """
        Enable mouse wheel scrolling.

        This function enables scrolling through the results using the mouse wheel,
        improving the navigation experience for the user. It is bound to mouse wheel
        events to allow for smooth scrolling.

        Args:
            event (tk.Event): The event that triggered the function.

        Returns:
            None
        """
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", on_mouse_wheel)
    canvas.bind_all("<Button-4>", on_mouse_wheel)
    canvas.bind_all("<Button-5>", on_mouse_wheel)

    root.mainloop()

if __name__ == "__main__":
    start_ui(fetch_and_display_results)
