"""
Enhanced Tika PDF Extraction with Better Table, Math, and Structure Handling

This module provides improved extraction of:
- Tables (preserved structure with delimiters)
- Mathematical equations (LaTeX patterns preserved)
- Document structure (sections, figures, captions)
- References and citations
"""

from tika import parser
import os
import json
import tika
import re
from typing import Dict, List, Optional, Tuple

tika.initVM()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNSTRUCTURED_DIR = os.path.join(BASE_DIR, "pdfs")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "tika_extracted_enhanced.jsonl")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# =============================================================================
# MATH AND EQUATION PATTERNS
# =============================================================================

# Patterns to detect and preserve mathematical content
MATH_PATTERNS = {
    # LaTeX inline math
    'latex_inline': r'\$[^$]+\$',
    # LaTeX display math
    'latex_display': r'\$\$[^$]+\$\$',
    # Common math operators and symbols
    'operators': r'[∝∞∑∏∫∂∇√±×÷≤≥≠≈≡∈∉⊂⊃∪∩∧∨¬∀∃αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ]',
    # Superscripts/subscripts patterns
    'subscript_superscript': r'[₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ₐₑₒₓₔₕₖₗₘₙₚₛₜ]',
    # Fraction-like patterns
    'fractions': r'\d+/\d+|[½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]',
    # Equation indicators
    'equation_markers': r'\((?:eq\.|Eq\.|equation|Equation)\s*\d+[a-z]?\)',
    # Common mathematical expressions
    'math_expressions': r'(?:sin|cos|tan|log|ln|exp|lim|max|min|sup|inf|det|dim|ker|im|arg)\s*\(',
}

# Combined regex for math detection
MATH_DETECTION_REGEX = re.compile(
    '|'.join(f'({pattern})' for pattern in MATH_PATTERNS.values()),
    re.UNICODE
)


# =============================================================================
# TABLE DETECTION AND EXTRACTION
# =============================================================================

def detect_table_regions(text: str) -> List[Dict]:
    """
    Detect potential table regions in text based on patterns.
    Returns list of table regions with start/end positions.
    """
    tables = []
    
    # Pattern 1: Lines with multiple tab/space separations suggesting columns
    lines = text.split('\n')
    table_start = None
    table_lines = []
    
    for i, line in enumerate(lines):
        # Check if line looks like a table row (multiple column-like separations)
        # Count whitespace-separated "cells"
        cells = re.split(r'\s{2,}|\t', line.strip())
        cells = [c for c in cells if c.strip()]
        
        # Heuristic: line with 3+ cells that are short might be table row
        is_table_row = (
            len(cells) >= 2 and 
            all(len(c) < 50 for c in cells) and
            len(line.strip()) > 0
        )
        
        if is_table_row:
            if table_start is None:
                table_start = i
            table_lines.append(line)
        else:
            # End of potential table
            if table_start is not None and len(table_lines) >= 2:
                tables.append({
                    'start_line': table_start,
                    'end_line': i - 1,
                    'content': '\n'.join(table_lines),
                    'num_rows': len(table_lines)
                })
            table_start = None
            table_lines = []
    
    # Handle table at end of document
    if table_start is not None and len(table_lines) >= 2:
        tables.append({
            'start_line': table_start,
            'end_line': len(lines) - 1,
            'content': '\n'.join(table_lines),
            'num_rows': len(table_lines)
        })
    
    return tables


def extract_tables_from_xml(xml_content: str) -> List[Dict]:
    """
    Extract tables from Tika's XML/HTML output.
    """
    tables = []
    
    # Pattern for HTML table elements
    table_pattern = re.compile(r'<table[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
    
    for match in table_pattern.finditer(xml_content):
        table_html = match.group(0)
        table_content = match.group(1)
        
        # Extract rows
        rows = []
        row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
        cell_pattern = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL | re.IGNORECASE)
        
        for row_match in row_pattern.finditer(table_content):
            cells = []
            for cell_match in cell_pattern.finditer(row_match.group(1)):
                # Clean cell content
                cell_text = re.sub(r'<[^>]+>', '', cell_match.group(1))
                cell_text = cell_text.strip()
                cells.append(cell_text)
            if cells:
                rows.append(cells)
        
        if rows:
            tables.append({
                'rows': rows,
                'num_rows': len(rows),
                'num_cols': max(len(row) for row in rows) if rows else 0,
                'markdown': convert_table_to_markdown(rows)
            })
    
    return tables


def convert_table_to_markdown(rows: List[List[str]]) -> str:
    """Convert table rows to markdown format."""
    if not rows:
        return ""
    
    markdown_lines = []
    
    # Header row
    if rows:
        header = ' | '.join(rows[0])
        markdown_lines.append(f"| {header} |")
        # Separator
        separator = ' | '.join(['---'] * len(rows[0]))
        markdown_lines.append(f"| {separator} |")
    
    # Data rows
    for row in rows[1:]:
        # Pad row if needed
        while len(row) < len(rows[0]):
            row.append('')
        data = ' | '.join(row[:len(rows[0])])
        markdown_lines.append(f"| {data} |")
    
    return '\n'.join(markdown_lines)


# =============================================================================
# MATHEMATICAL EQUATION EXTRACTION
# =============================================================================

def extract_equations(text: str) -> List[Dict]:
    """
    Extract mathematical equations from text.
    """
    equations = []
    
    # Pattern 1: Numbered equations like (1), (2a), Eq. 1, etc.
    eq_pattern = re.compile(
        r'(?:^|\n)\s*(.{10,200}?)\s*\((?:eq\.?|Eq\.?)?\s*(\d+[a-z]?)\)',
        re.MULTILINE
    )
    
    for match in eq_pattern.finditer(text):
        eq_content = match.group(1).strip()
        eq_number = match.group(2)
        
        # Check if it looks like an equation (contains math-like content)
        if MATH_DETECTION_REGEX.search(eq_content) or '=' in eq_content:
            equations.append({
                'content': eq_content,
                'number': eq_number,
                'type': 'numbered'
            })
    
    # Pattern 2: Lines that look like standalone equations
    for line in text.split('\n'):
        line = line.strip()
        if len(line) > 5 and len(line) < 200:
            # Check for equation-like content
            math_chars = sum(1 for c in line if c in '=∝∞∑∏∫∂∇√±×÷≤≥≠≈≡+-*/^()')
            letter_count = sum(1 for c in line if c.isalpha())
            
            # Heuristic: high ratio of math chars to letters suggests equation
            if letter_count > 0 and math_chars / letter_count > 0.3:
                if line not in [eq['content'] for eq in equations]:
                    equations.append({
                        'content': line,
                        'number': None,
                        'type': 'inline'
                    })
    
    return equations


def preserve_latex_patterns(text: str) -> str:
    """
    Preserve LaTeX patterns that might get mangled during extraction.
    """
    # Common LaTeX command patterns
    latex_commands = [
        (r'\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}', r'((\1)/(\2))'),  # fractions
        (r'\\sqrt\s*\{([^}]+)\}', r'√(\1)'),  # square root
        (r'\\sum', '∑'),
        (r'\\prod', '∏'),
        (r'\\int', '∫'),
        (r'\\partial', '∂'),
        (r'\\nabla', '∇'),
        (r'\\infty', '∞'),
        (r'\\alpha', 'α'),
        (r'\\beta', 'β'),
        (r'\\gamma', 'γ'),
        (r'\\delta', 'δ'),
        (r'\\epsilon', 'ε'),
        (r'\\theta', 'θ'),
        (r'\\lambda', 'λ'),
        (r'\\mu', 'μ'),
        (r'\\pi', 'π'),
        (r'\\sigma', 'σ'),
        (r'\\phi', 'φ'),
        (r'\\omega', 'ω'),
        (r'\\times', '×'),
        (r'\\cdot', '·'),
        (r'\\pm', '±'),
        (r'\\leq', '≤'),
        (r'\\geq', '≥'),
        (r'\\neq', '≠'),
        (r'\\approx', '≈'),
        (r'\\equiv', '≡'),
        (r'\\propto', '∝'),
        (r'\^(\d+)', r'<sup>\1</sup>'),  # superscripts
        (r'_(\d+)', r'<sub>\1</sub>'),   # subscripts
    ]
    
    for pattern, replacement in latex_commands:
        text = re.sub(pattern, replacement, text)
    
    return text


# =============================================================================
# STRUCTURE EXTRACTION
# =============================================================================

def extract_sections(text: str) -> List[Dict]:
    """
    Extract document sections (headers, abstract, references, etc.)
    """
    sections = []
    
    # Common section headers in academic papers
    section_patterns = [
        r'^(?:I{1,3}V?|IV|V?I{0,3})\.?\s+([A-Z][A-Z\s]+)$',  # Roman numerals
        r'^(\d+\.?\s+[A-Z][A-Z\s]+)$',  # Numbered sections
        r'^(ABSTRACT|Abstract|INTRODUCTION|Introduction|METHODS?|Methods?|RESULTS?|Results?|DISCUSSION|Discussion|CONCLUSION|Conclusion|REFERENCES|References|ACKNOWLEDGMENT|Acknowledgment)s?:?\s*$',
    ]
    
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        for pattern in section_patterns:
            match = re.match(pattern, line_stripped)
            if match:
                sections.append({
                    'title': line_stripped,
                    'line_number': i,
                    'type': 'section_header'
                })
                break
    
    return sections


def extract_figures_and_captions(text: str) -> List[Dict]:
    """
    Extract figure and table captions.
    """
    figures = []
    
    # Figure/Table caption patterns
    patterns = [
        r'(?:Fig(?:ure)?|FIG(?:URE)?)\s*\.?\s*(\d+[a-z]?)\s*[:\.]?\s*(.{10,300})',
        r'(?:Table|TABLE)\s*\.?\s*(\d+[a-z]?)\s*[:\.]?\s*(.{10,300})',
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            fig_type = 'figure' if 'fig' in match.group(0).lower() else 'table'
            figures.append({
                'type': fig_type,
                'number': match.group(1),
                'caption': match.group(2).strip(),
            })
    
    return figures


def extract_references(text: str) -> List[str]:
    """
    Extract reference entries from the document.
    """
    references = []
    
    # Find the references section
    ref_start = None
    patterns = [
        r'(?:^|\n)\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)\s*(?:\n|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            ref_start = match.end()
            break
    
    if ref_start:
        ref_text = text[ref_start:]
        
        # Pattern for numbered references [1], 1., etc.
        ref_patterns = [
            r'\[(\d+)\]\s*([^\[\]]{20,500}?)(?=\[\d+\]|\n\n|$)',
            r'^(\d+)\.\s*([^\n]{20,500})',
        ]
        
        for pattern in ref_patterns:
            for match in re.finditer(pattern, ref_text, re.MULTILINE):
                references.append({
                    'number': match.group(1),
                    'text': match.group(2).strip()
                })
    
    return references


# =============================================================================
# IMPROVED TEXT CLEANING
# =============================================================================

def clean_text_enhanced(text: str, preserve_structure: bool = True) -> str:
    """
    Enhanced text cleaning that preserves important formatting.
    """
    if not text:
        return ""
    
    # Preserve LaTeX patterns first
    text = preserve_latex_patterns(text)
    
    # Remove header/footer artifacts (page numbers, running headers)
    text = re.sub(r'^[\d\s]+$', '', text, flags=re.MULTILINE)  # standalone page numbers
    
    # Fix common OCR/extraction issues
    text = re.sub(r'(?<=[a-z])-\s*\n\s*(?=[a-z])', '', text)  # hyphenation at line breaks
    
    if preserve_structure:
        # Normalize multiple newlines but keep paragraph structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Normalize spaces within lines
        text = re.sub(r'[ \t]+', ' ', text)
    else:
        # Collapse all whitespace for dense representation
        text = re.sub(r'\s+', ' ', text)
    
    # Clean up arXiv-style header artifacts
    text = re.sub(r'ar\s*X\s*iv\s*:\s*[\d.]+v?\d*\s*\[[^\]]+\]\s*\d+\s*\w+\s*\d+', '', text)
    
    return text.strip()


def extract_abstract(text: str) -> Optional[str]:
    """
    Extract the abstract from the document.
    """
    patterns = [
        r'(?:ABSTRACT|Abstract)\s*[:\-]?\s*(.*?)(?=\n\s*(?:I\.?\s+)?(?:INTRODUCTION|Introduction|Keywords|KEYWORDS|1\.\s+Introduction))',
        r'(?:ABSTRACT|Abstract)\s*[:\-]?\s*(.*?)(?=\n\n)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            abstract = match.group(1).strip()
            # Clean up the abstract
            abstract = re.sub(r'\s+', ' ', abstract)
            if len(abstract) > 50:  # Reasonable abstract length
                return abstract
    
    return None


# =============================================================================
# MAIN EXTRACTION WITH XML OUTPUT
# =============================================================================

def extract_with_xml(filepath: str) -> Tuple[str, str, Dict]:
    """
    Extract content using Tika's XML output for better structure preservation.
    """
    # Get XML/HTML output for structure
    parsed_xml = parser.from_file(filepath, xmlContent=True)
    xml_content = parsed_xml.get("content", "")
    
    # Get plain text for reading
    parsed_text = parser.from_file(filepath)
    text_content = parsed_text.get("content", "")
    metadata = parsed_text.get("metadata", {})
    
    return text_content, xml_content, metadata


def process_pdf(filepath: str, filename: str) -> Dict:
    """
    Process a single PDF with enhanced extraction.
    """
    paper_id = filename.replace(".pdf", "")
    
    try:
        # Extract content
        text_content, xml_content, metadata = extract_with_xml(filepath)
        
        # Clean text while preserving structure
        clean_full_text = clean_text_enhanced(text_content, preserve_structure=True)
        clean_dense_text = clean_text_enhanced(text_content, preserve_structure=False)
        
        # Extract structured elements
        tables_from_text = detect_table_regions(text_content)
        tables_from_xml = extract_tables_from_xml(xml_content) if xml_content else []
        equations = extract_equations(clean_full_text)
        sections = extract_sections(clean_full_text)
        figures = extract_figures_and_captions(clean_full_text)
        references = extract_references(clean_full_text)
        abstract = extract_abstract(clean_full_text)
        
        # Combine tables from both sources
        all_tables = tables_from_xml if tables_from_xml else tables_from_text
        
        # Build comprehensive record
        record = {
            # Alignment keys
            "paper_id": paper_id,
            "file_name": filename,
            
            # Text content
            "full_text": clean_full_text,
            "dense_text": clean_dense_text,  # Single-line version for embeddings
            "abstract": abstract,
            
            # Structured elements
            "sections": sections,
            "tables": all_tables,
            "equations": equations,
            "figures": figures,
            "references": references,
            
            # Counts
            "num_tables": len(all_tables),
            "num_equations": len(equations),
            "num_figures": len(figures),
            "num_references": len(references),
            "num_sections": len(sections),
            
            # PDF metadata
            "num_pages": metadata.get("xmpTPg:NPages"),
            "language": metadata.get("language"),
            "content_type": metadata.get("Content-Type"),
            "created": metadata.get("Creation-Date"),
            "producer": metadata.get("pdf:producer"),
            "title": metadata.get("dc:title") or metadata.get("title"),
            "author": metadata.get("dc:creator") or metadata.get("Author"),
            
            # Statistics
            "char_count": len(clean_full_text),
            "word_count": len(clean_full_text.split()),
            
            # Quality indicators
            "has_math": len(equations) > 0 or bool(MATH_DETECTION_REGEX.search(clean_full_text)),
            "has_tables": len(all_tables) > 0,
            "extraction_quality": "enhanced"
        }
        
        return record
        
    except Exception as e:
        return {
            "paper_id": paper_id,
            "file_name": filename,
            "error": str(e),
            "extraction_quality": "failed"
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main extraction loop."""
    print("=" * 60)
    print("Enhanced Tika PDF Extraction")
    print("=" * 60)
    
    pdf_files = [f for f in os.listdir(UNSTRUCTURED_DIR) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files to process\n")
    
    stats = {
        'total': len(pdf_files),
        'success': 0,
        'failed': 0,
        'with_tables': 0,
        'with_equations': 0,
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, filename in enumerate(pdf_files, 1):
            filepath = os.path.join(UNSTRUCTURED_DIR, filename)
            print(f"[{i}/{len(pdf_files)}] Processing: {filename}")
            
            record = process_pdf(filepath, filename)
            
            if 'error' not in record:
                stats['success'] += 1
                if record.get('has_tables'):
                    stats['with_tables'] += 1
                if record.get('has_math'):
                    stats['with_equations'] += 1
                print(f"  ✓ Tables: {record['num_tables']}, Equations: {record['num_equations']}, "
                      f"Sections: {record['num_sections']}, Words: {record['word_count']}")
            else:
                stats['failed'] += 1
                print(f"  ✗ Error: {record['error']}")
            
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print("=" * 60)
    print(f"Total PDFs:      {stats['total']}")
    print(f"Successful:      {stats['success']}")
    print(f"Failed:          {stats['failed']}")
    print(f"With Tables:     {stats['with_tables']}")
    print(f"With Math/Eqs:   {stats['with_equations']}")
    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
