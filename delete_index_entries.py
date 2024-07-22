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
    


if __name__ == "__main__":
    doc_ids = [7435]
    delete_docs_by_id(doc_ids)