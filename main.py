import UI
import threading
from retrieval import main_retrival

def fetch_and_display_results(query, use_spellcheck=True):
    old_query, spellchecked_query, ranked_documents = main_retrival(query, need_spellcheck=use_spellcheck, index_db="index_with_position.db")
    
    if old_query.lower() != spellchecked_query.lower() and use_spellcheck:
        # If the query was corrected and spellcheck was used, prompt the user
        UI.display_corrected_query(old_query, spellchecked_query)
        query = spellchecked_query
    else:
        query = old_query

    # Prepare results for the UI
    results = []
    for i, (doc_id, score, url, name, topics, snippet) in enumerate(ranked_documents):
        if not isinstance(topics, list):
            topics = [topics]  
        topics = [topic for topic in topics if topic]  
        results.append({
            "Name of site": name,
            "Url": url,
            "Keywords": ', '.join(topics),  
            "Preview": snippet  
        })

    # Update the UI with the results
    UI.update_results(query, [], results)

def main():
    # Start the UI in a separate thread
    ui_thread = threading.Thread(target=UI.start_ui, args=(fetch_and_display_results,))
    ui_thread.start()

    # Keep the main thread alive while the UI is running
    ui_thread.join()

if __name__ == "__main__":
    main()
