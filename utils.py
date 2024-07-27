from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_cv(cv_row):
    """
    This function is needed to chunck the column text in a row and keep the other data into metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=20)
    chunks = []
    for i, chunk in enumerate(text_splitter.split_text(cv_row["text"])):
        chunks.append(
            {
                "id": f"{cv_row['id']}chunk{i}",  # Unique chunk ID
                "text": chunk,
                "metadata": {
                    "cv_id": cv_row["id"],
                    "name": cv_row["name"],
                    "email": cv_row["email"],
                    "skills": cv_row["skills"],
                    "experience": cv_row["experience"],
                    "chunk_id": i,
                },
            }
        )
    return chunks