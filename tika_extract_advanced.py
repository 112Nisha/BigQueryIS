"""
Advanced PDF Extraction using PyMuPDF (fitz) + Tika Hybrid Approach

This provides superior extraction for:
- Tables (using pymupdf's table detection)
- Mathematical equations (better Unicode handling)
- Document structure (blocks, spans, fonts)
- Images and figures

Install dependencies:
    pip install pymupdf tika

For even better table extraction, optionally install:
    pip install camelot-py[cv] tabula-py
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not installed. Run: pip install pymupdf")

try:
    from tika import parser
    import tika
    tika.initVM()
    HAS_TIKA = True
except ImportError:
    HAS_TIKA = False
    print("Warning: Tika not installed. Run: pip install tika")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNSTRUCTURED_DIR = os.path.join(BASE_DIR, "pdfs")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "tika_extracted_advanced.jsonl")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TableCell:
    text: str
    row: int
    col: int
    colspan: int = 1
    rowspan: int = 1

@dataclass  
class ExtractedTable:
    page: int
    rows: List[List[str]]
    bbox: Optional[Tuple[float, float, float, float]] = None
    markdown: str = ""
    
@dataclass
class ExtractedEquation:
    content: str
    page: int
    number: Optional[str] = None
    latex: Optional[str] = None
    confidence: float = 0.0

@dataclass
class ExtractedFigure:
    page: int
    caption: str
    number: str
    figure_type: str  # 'figure' or 'table'
    bbox: Optional[Tuple[float, float, float, float]] = None


# =============================================================================
# PYMUPDF EXTRACTION (SUPERIOR FOR STRUCTURE)
# =============================================================================

class PyMuPDFExtractor:
    """Extract content using PyMuPDF for better structure preservation."""
    
    # Mathematical and scientific Unicode ranges
    MATH_UNICODE_RANGES = [
        (0x0370, 0x03FF),  # Greek
        (0x2100, 0x214F),  # Letterlike Symbols
        (0x2150, 0x218F),  # Number Forms
        (0x2190, 0x21FF),  # Arrows
        (0x2200, 0x22FF),  # Mathematical Operators
        (0x2300, 0x23FF),  # Miscellaneous Technical
        (0x27C0, 0x27EF),  # Miscellaneous Mathematical Symbols-A
        (0x2980, 0x29FF),  # Miscellaneous Mathematical Symbols-B
        (0x2A00, 0x2AFF),  # Supplemental Mathematical Operators
        (0x1D400, 0x1D7FF),  # Mathematical Alphanumeric Symbols
    ]
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.doc = fitz.open(filepath)
        
    def close(self):
        self.doc.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        self.close()
    
    def extract_text_with_structure(self) -> Dict[str, Any]:
        """Extract text preserving document structure."""
        pages_text = []
        all_blocks = []
        
        for page_num, page in enumerate(self.doc):
            # Get text with detailed block info
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            
            page_text = []
            for block in blocks.get("blocks", []):
                if block["type"] == 0:  # Text block
                    block_text = []
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            text = span.get("text", "")
                            font = span.get("font", "")
                            size = span.get("size", 12)
                            
                            # Detect if this might be math based on font
                            is_math_font = any(x in font.lower() for x in 
                                             ['math', 'symbol', 'cmex', 'cmsy', 'cmmi', 'cmr'])
                            
                            if is_math_font:
                                text = f"[MATH:{text}]"
                            
                            line_text += text
                        block_text.append(line_text)
                    
                    full_block = "\n".join(block_text)
                    page_text.append(full_block)
                    
                    all_blocks.append({
                        "page": page_num,
                        "bbox": block.get("bbox"),
                        "text": full_block,
                        "type": "text"
                    })
            
            pages_text.append("\n\n".join(page_text))
        
        return {
            "full_text": "\n\n---PAGE BREAK---\n\n".join(pages_text),
            "pages": pages_text,
            "blocks": all_blocks,
            "num_pages": len(self.doc)
        }
    
    def extract_tables(self) -> List[ExtractedTable]:
        """Extract tables using PyMuPDF's table detection."""
        tables = []
        
        for page_num, page in enumerate(self.doc):
            # Try to find tables on this page
            try:
                # PyMuPDF 1.23+ has built-in table detection
                if hasattr(page, 'find_tables'):
                    page_tables = page.find_tables()
                    for tab in page_tables:
                        # Extract table data
                        rows = []
                        for row in tab.extract():
                            cleaned_row = [cell if cell else "" for cell in row]
                            rows.append(cleaned_row)
                        
                        if rows and any(any(cell for cell in row) for row in rows):
                            tables.append(ExtractedTable(
                                page=page_num,
                                rows=rows,
                                bbox=tab.bbox if hasattr(tab, 'bbox') else None,
                                markdown=self._rows_to_markdown(rows)
                            ))
                else:
                    # Fallback: detect tables using text positioning
                    tables.extend(self._detect_tables_by_position(page, page_num))
                    
            except Exception as e:
                print(f"  Warning: Table extraction failed on page {page_num}: {e}")
                continue
        
        return tables
    
    def _detect_tables_by_position(self, page, page_num: int) -> List[ExtractedTable]:
        """Detect tables by analyzing text block positions."""
        tables = []
        blocks = page.get_text("dict")
        
        # Group blocks by Y position (rows)
        y_groups = defaultdict(list)
        for block in blocks.get("blocks", []):
            if block["type"] == 0:
                y = round(block["bbox"][1], 0)  # Top Y coordinate
                y_groups[y].append(block)
        
        # Find rows with multiple aligned columns
        potential_table_rows = []
        for y, row_blocks in sorted(y_groups.items()):
            if len(row_blocks) >= 2:
                # Extract text from each block
                cells = []
                for block in sorted(row_blocks, key=lambda b: b["bbox"][0]):
                    cell_text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            cell_text += span.get("text", "")
                    cells.append(cell_text.strip())
                
                if cells and any(cells):
                    potential_table_rows.append((y, cells))
        
        # Group consecutive rows into tables
        if potential_table_rows:
            current_table = [potential_table_rows[0][1]]
            prev_y = potential_table_rows[0][0]
            
            for y, cells in potential_table_rows[1:]:
                if y - prev_y < 30:  # Within same table
                    current_table.append(cells)
                else:
                    if len(current_table) >= 2:
                        tables.append(ExtractedTable(
                            page=page_num,
                            rows=current_table,
                            markdown=self._rows_to_markdown(current_table)
                        ))
                    current_table = [cells]
                prev_y = y
            
            if len(current_table) >= 2:
                tables.append(ExtractedTable(
                    page=page_num,
                    rows=current_table,
                    markdown=self._rows_to_markdown(current_table)
                ))
        
        return tables
    
    def _rows_to_markdown(self, rows: List[List[str]]) -> str:
        """Convert table rows to markdown."""
        if not rows:
            return ""
        
        # Normalize column count
        max_cols = max(len(row) for row in rows)
        normalized = []
        for row in rows:
            normalized.append(row + [''] * (max_cols - len(row)))
        
        lines = []
        # Header
        lines.append("| " + " | ".join(normalized[0]) + " |")
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        # Data
        for row in normalized[1:]:
            lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)
    
    def extract_equations(self) -> List[ExtractedEquation]:
        """Extract mathematical equations."""
        equations = []
        
        for page_num, page in enumerate(self.doc):
            text = page.get_text()
            
            # Pattern 1: Numbered equations
            eq_patterns = [
                r'([^\n]{10,150}?)\s*\((\d+[a-z]?)\)\s*$',
                r'(?:Eq(?:uation)?\.?\s*)?(\d+[a-z]?):\s*([^\n]{10,150})',
            ]
            
            for pattern in eq_patterns:
                for match in re.finditer(pattern, text, re.MULTILINE):
                    content = match.group(1) if pattern.startswith('(') else match.group(2)
                    number = match.group(2) if pattern.startswith('(') else match.group(1)
                    
                    # Check if it looks like an equation
                    if self._looks_like_equation(content):
                        equations.append(ExtractedEquation(
                            content=content.strip(),
                            page=page_num,
                            number=number,
                            confidence=self._equation_confidence(content)
                        ))
            
            # Pattern 2: Detect by font/formatting
            blocks = page.get_text("dict")
            for block in blocks.get("blocks", []):
                if block["type"] == 0:
                    for line in block.get("lines", []):
                        line_text = ""
                        is_math = False
                        for span in line.get("spans", []):
                            text = span.get("text", "")
                            font = span.get("font", "")
                            
                            # Math fonts
                            if any(x in font.lower() for x in 
                                  ['math', 'symbol', 'cmex', 'cmsy', 'cmmi']):
                                is_math = True
                            
                            line_text += text
                        
                        if is_math and len(line_text.strip()) > 3:
                            # Avoid duplicates
                            if not any(eq.content == line_text.strip() for eq in equations):
                                equations.append(ExtractedEquation(
                                    content=line_text.strip(),
                                    page=page_num,
                                    confidence=0.8
                                ))
        
        return equations
    
    def _looks_like_equation(self, text: str) -> bool:
        """Check if text looks like a mathematical equation."""
        # Must have equals sign or mathematical operators
        math_chars = set('=∝∞∑∏∫∂∇√±×÷≤≥≠≈≡∈∉⊂⊃∪∩+-*/^')
        has_math = any(c in math_chars for c in text)
        
        # Should have some letters (variables)
        has_vars = bool(re.search(r'[a-zA-Zαβγδεζηθικλμνξπρστυφχψω]', text))
        
        # Not too long (probably a sentence)
        not_too_long = len(text) < 200
        
        return has_math and has_vars and not_too_long
    
    def _equation_confidence(self, text: str) -> float:
        """Calculate confidence that text is an equation."""
        score = 0.0
        
        # Presence of equals sign
        if '=' in text:
            score += 0.3
        
        # Greek letters
        if re.search(r'[αβγδεζηθικλμνξπρστυφχψω]', text, re.IGNORECASE):
            score += 0.2
        
        # Mathematical operators
        math_ops = set('∑∏∫∂∇√±×÷≤≥≠≈≡∝')
        if any(c in math_ops for c in text):
            score += 0.3
        
        # Subscripts/superscripts patterns
        if re.search(r'[_^]\{?[^}]+\}?|\d+', text):
            score += 0.1
        
        # Fractions
        if '/' in text or re.search(r'\\frac', text):
            score += 0.1
        
        return min(score, 1.0)
    
    def extract_figures_and_captions(self) -> List[ExtractedFigure]:
        """Extract figure and table captions."""
        figures = []
        
        for page_num, page in enumerate(self.doc):
            text = page.get_text()
            
            patterns = [
                (r'(?:Fig(?:ure)?|FIG(?:URE)?)\s*\.?\s*(\d+[a-z]?)\s*[:\.]?\s*(.{10,300})', 'figure'),
                (r'(?:Table|TABLE)\s*\.?\s*(\d+[a-z]?)\s*[:\.]?\s*(.{10,300})', 'table'),
            ]
            
            for pattern, fig_type in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    figures.append(ExtractedFigure(
                        page=page_num,
                        number=match.group(1),
                        caption=match.group(2).strip()[:300],
                        figure_type=fig_type
                    ))
        
        return figures
    
    def get_metadata(self) -> Dict:
        """Extract PDF metadata."""
        meta = self.doc.metadata or {}
        return {
            "title": meta.get("title"),
            "author": meta.get("author"),
            "subject": meta.get("subject"),
            "creator": meta.get("creator"),
            "producer": meta.get("producer"),
            "created": meta.get("creationDate"),
            "modified": meta.get("modDate"),
            "num_pages": len(self.doc)
        }


# =============================================================================
# HYBRID EXTRACTION (BEST OF BOTH WORLDS)
# =============================================================================

def extract_hybrid(filepath: str, filename: str) -> Dict:
    """
    Hybrid extraction using PyMuPDF for structure and Tika for text.
    """
    paper_id = filename.replace(".pdf", "")
    result = {
        "paper_id": paper_id,
        "file_name": filename,
    }
    
    try:
        # PyMuPDF extraction (structure)
        if HAS_PYMUPDF:
            with PyMuPDFExtractor(filepath) as extractor:
                # Text with structure
                text_data = extractor.extract_text_with_structure()
                result["full_text"] = text_data["full_text"]
                result["pages_text"] = text_data["pages"]
                result["num_pages"] = text_data["num_pages"]
                
                # Tables
                tables = extractor.extract_tables()
                result["tables"] = [asdict(t) for t in tables]
                result["num_tables"] = len(tables)
                
                # Equations
                equations = extractor.extract_equations()
                result["equations"] = [asdict(e) for e in equations]
                result["num_equations"] = len(equations)
                
                # Figures
                figures = extractor.extract_figures_and_captions()
                result["figures"] = [asdict(f) for f in figures]
                result["num_figures"] = len(figures)
                
                # Metadata
                meta = extractor.get_metadata()
                result.update({
                    "title": meta.get("title"),
                    "author": meta.get("author"),
                    "producer": meta.get("producer"),
                    "created": meta.get("created"),
                })
        
        # Tika extraction (backup/comparison)
        if HAS_TIKA:
            parsed = parser.from_file(filepath)
            tika_text = parsed.get("content", "")
            tika_meta = parsed.get("metadata", {})
            
            # If PyMuPDF failed, use Tika
            if "full_text" not in result or not result["full_text"]:
                result["full_text"] = tika_text
            
            # Merge metadata
            if not result.get("title"):
                result["title"] = tika_meta.get("dc:title") or tika_meta.get("title")
            if not result.get("author"):
                result["author"] = tika_meta.get("dc:creator") or tika_meta.get("Author")
            if not result.get("num_pages"):
                result["num_pages"] = tika_meta.get("xmpTPg:NPages")
            
            result["language"] = tika_meta.get("language")
            result["content_type"] = tika_meta.get("Content-Type")
        
        # Clean text
        full_text = result.get("full_text", "")
        result["full_text"] = clean_extracted_text(full_text)
        
        # Dense text for embeddings
        result["dense_text"] = re.sub(r'\s+', ' ', result["full_text"]).strip()
        
        # Extract abstract
        result["abstract"] = extract_abstract(result["full_text"])
        
        # Statistics
        result["char_count"] = len(result["full_text"])
        result["word_count"] = len(result["full_text"].split())
        result["has_math"] = result.get("num_equations", 0) > 0
        result["has_tables"] = result.get("num_tables", 0) > 0
        result["extraction_method"] = "hybrid" if (HAS_PYMUPDF and HAS_TIKA) else \
                                      ("pymupdf" if HAS_PYMUPDF else "tika")
        
    except Exception as e:
        result["error"] = str(e)
        result["extraction_method"] = "failed"
    
    return result


def clean_extracted_text(text: str) -> str:
    """Clean extracted text while preserving structure."""
    if not text:
        return ""
    
    # Remove page break markers for dense text
    text = re.sub(r'---PAGE BREAK---', '\n\n', text)
    
    # Fix hyphenation at line breaks
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    
    # Remove arXiv header artifacts
    text = re.sub(r'ar\s*X\s*iv\s*:\s*[\d.]+v?\d*\s*\[[^\]]+\]\s*\d+\s*\w+\s*\d+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def extract_abstract(text: str) -> Optional[str]:
    """Extract abstract from document."""
    patterns = [
        r'(?:ABSTRACT|Abstract)\s*[:\-]?\s*(.*?)(?=\n\s*(?:I\.?\s+)?(?:INTRODUCTION|Introduction|Keywords|1\.\s+Introduction))',
        r'(?:ABSTRACT|Abstract)\s*[:\-]?\s*\n(.*?)(?=\n\n)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            abstract = re.sub(r'\s+', ' ', match.group(1)).strip()
            if 50 < len(abstract) < 3000:
                return abstract
    
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Advanced PDF Extraction (PyMuPDF + Tika Hybrid)")
    print("=" * 60)
    
    if not HAS_PYMUPDF and not HAS_TIKA:
        print("ERROR: Neither PyMuPDF nor Tika is installed!")
        print("Install with: pip install pymupdf tika")
        return
    
    print(f"PyMuPDF: {'✓' if HAS_PYMUPDF else '✗'}")
    print(f"Tika: {'✓' if HAS_TIKA else '✗'}")
    print()
    
    pdf_files = [f for f in os.listdir(UNSTRUCTURED_DIR) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files\n")
    
    stats = {'success': 0, 'failed': 0, 'tables': 0, 'equations': 0}
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, filename in enumerate(pdf_files, 1):
            filepath = os.path.join(UNSTRUCTURED_DIR, filename)
            print(f"[{i}/{len(pdf_files)}] {filename}")
            
            record = extract_hybrid(filepath, filename)
            
            if 'error' not in record:
                stats['success'] += 1
                stats['tables'] += record.get('num_tables', 0)
                stats['equations'] += record.get('num_equations', 0)
                print(f"  ✓ Tables: {record.get('num_tables', 0)}, "
                      f"Equations: {record.get('num_equations', 0)}, "
                      f"Words: {record.get('word_count', 0)}")
            else:
                stats['failed'] += 1
                print(f"  ✗ {record['error']}")
            
            out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    
    print("\n" + "=" * 60)
    print("Complete!")
    print(f"Success: {stats['success']}, Failed: {stats['failed']}")
    print(f"Total Tables: {stats['tables']}, Total Equations: {stats['equations']}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
