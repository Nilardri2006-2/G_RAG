from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150
    )

    chunks = []
    for page in pages:
        if not page["text"].strip():
            continue
        page_chunks = splitter.split_text(page["text"])
        for chunk in page_chunks:
            chunks.append({
                "text" : chunk,
                "page" : page["page"],
                "source": "Constitution of India",
                "document_type": "constitution"
            })
    return chunks