import sqlite3
def delete_docs_by_id(doc_ids, db_name='index_with_position.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create a placeholder string for the number of doc_ids
    placeholders = ','.join('?' for _ in doc_ids)

    # Delete from Positions
    cursor.execute(f'DELETE FROM Positions WHERE doc_id IN ({placeholders})', doc_ids)

    # Delete from Documents
    cursor.execute(f'DELETE FROM Documents WHERE doc_id IN ({placeholders})', doc_ids)

    conn.commit()
    conn.close()

def get_matching_ids(url_substring, db_name='search.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Select IDs where the URL contains the provided substring
    cursor.execute('SELECT rowid FROM pages WHERE url LIKE ?', (f"%{url_substring}%",))
    rows = cursor.fetchall()

    # Extract IDs from the rows
    matching_ids = [row[0] for row in rows]

    conn.close()
    return matching_ids
    


if __name__ == "__main__":
    #doc_ids = [7476, 7591, 7566, 7565, 7581, 6363, 7546, 7587, 7582, 7531, 7394, 7202, 6392, 6290, 7218, 7402, 7275, 7375, 7374, 169, 2713, 81, 109, 197, 198, 7479, 7238, 7274]
    doc_ids = get_matching_ids("https://www.kreis-tuebingen.de")
    doc_ids.sort()
    print(doc_ids)
    #doc_ids = [5875]
    delete_docs_by_id(doc_ids)