import fitz  # PyMuPDF
from PIL import Image
import io
import os
import logging
from typing import List, Dict, Any

log = logging.getLogger("pdf_parser")

class CatalogParser:
    def __init__(self, pdf_path: str, output_image_dir: str = "raw_data/images"):
        self.pdf_path = pdf_path
        self.output_image_dir = output_image_dir
        os.makedirs(self.output_image_dir, exist_ok=True)
        self.doc = fitz.open(self.pdf_path)

    def extract_page_data(self, page_num: int) -> Dict[str, Any]:
        """Extracts text blocks, links, and images from a single page."""
        page = self.doc.load_page(page_num)
        
        # 1. Extract Text Blocks (layout aware)
        blocks = page.get_text("blocks") 
        text_blocks = []
        for b in blocks:
            # block format: (x0, y0, x1, y1, "text", block_no, block_type)
            if b[6] == 0: # Is text block
                text_blocks.append({
                    "bbox": (b[0], b[1], b[2], b[3]),
                    "text": b[4].strip()
                })

        # 2. Extract Images
        images_data = []
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = self.doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"page_{page_num}_img_{img_index}.{image_ext}"
            image_path = os.path.join(self.output_image_dir, image_filename)
            
            # Save image
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
                
            images_data.append({
                "xref": xref,
                "path": image_path,
                "ext": image_ext
            })

        return {
            "page_num": page_num,
            "text_blocks": text_blocks,
            "images": images_data
        }

    def parse_catalog(self) -> List[Dict[str, Any]]:
        """Parses the entire catalog page by page."""
        log.info(f"Starting parsing of {self.pdf_path} ({len(self.doc)} pages)")
        parsed_pages = []
        for page_num in range(len(self.doc)):
            page_data = self.extract_page_data(page_num)
            parsed_pages.append(page_data)
        log.info(f"Finished parsing {self.pdf_path}")
        return parsed_pages

    def close(self):
        self.doc.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage:
    # parser = CatalogParser("sample.pdf")
    # data = parser.parse_catalog()
    # print(data[0])
