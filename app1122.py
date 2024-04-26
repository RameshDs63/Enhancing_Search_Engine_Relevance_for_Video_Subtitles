import streamlit as st
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Connect to the SQLite database
def connect_db(database_path):
    conn = sqlite3.connect("data/chromadb1122.db")
    return conn

def get_top_10_unique_names(query, conn, model):
    c = conn.cursor()
    query_embedding = model.encode([query])[0]  # Use [0] to get the single embedding from the list
    similarities = []
    c.execute("SELECT * FROM dd")
    for row in c.fetchall():
        dd_num, dd_name, dd_content_chunks, dd_embeddings = row  # Assuming the fourth column is not needed
        try:
            embeddings = np.fromstring(dd_embeddings[1:-1], sep=', ')  # Parse as numpy array
            # Check if embeddings is a list of valid numbers
            if embeddings.size == 0:  # Skip empty embeddings
                continue
            similarity = cosine_similarity(query_embedding.reshape(1, -1), embeddings.reshape(1, -1))[0][0]
            similarities.append((dd_name, similarity))
        except Exception as e:
            print(f"Error processing embeddings for {dd_name}: {e}")
            continue
    c.close()
    
    sorted_names = [name for name, _ in sorted(similarities, key=lambda x: x[1], reverse=True)]
    unique_names = []
    for name in sorted_names:
        if name not in unique_names:
            unique_names.append(name)
            if len(unique_names) == 10:
                break
    
    return unique_names

# Main function to run the Streamlit app
def main():
    st.title('Movie Title Search Engine')
    st.header('Enter words & sentences related to movies!')
    query = st.text_input('Enter your query:')
    database_path = 'data/chromadb1122.db'
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if st.button('Search'):
        if query:
            try:
                conn = connect_db(database_path)
                top_10_unique_names = get_top_10_unique_names(query, conn, model)
                if not top_10_unique_names:
                    st.write('No matches found.')
                else:
                    st.write('Top 10 Unique Names:')
                    for i, name in enumerate(top_10_unique_names, start=1):
                        st.write(f"{i}. {name}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                conn.close()

if __name__ == '__main__':
    main()
