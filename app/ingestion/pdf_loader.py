from pathlib import Path
from pypdf import PdfReader

def load_pdf(file_path:str) ->str:
    """ extreact pdf"""
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDf not found:{pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages = []

    for p_num , page in enumerate(reader.pages , start = 1):
        text = page.extract_text() or ""

        pages.append({
            "page" : p_num,
            "text" :text
        })
    return pages

if __name__ == "__main__":
    pages = load_pdf("data/raw/c1.pdf")
    print(f"total pages: {len(pages)}")
    print("\n--- First page ---\n")
    print(pages[0]["text"][:2000])