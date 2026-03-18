"""Tree builder — orchestrates PDF extraction, API discovery, and ML scoring."""

from __future__ import annotations

import hashlib
import os
from typing import List, Set, Tuple
import requests
from sklearn import tree
from sympy import root

from citation_tree.cache import Cache
from citation_tree.clients import ArxivClient, OAClient, S2Client
from citation_tree.config import MAX_DEPTH, MAX_PAPERS, MIN_RELEVANCE, PDFS_DIR, RATE_LIMIT
from citation_tree.ml import compute_similarity, generate_improvement_explanation
from citation_tree.models import CitationTree, Paper
from citation_tree.pdf import download_pdf, extract_pdf
from citation_tree.text_utils import important_words, title_hash, titles_match
from citation_tree.ml import compute_similarity

# Builds a CitationTree from a given PDF
class TreeBuilder:

    def __init__(
        self,
        max_depth: int = MAX_DEPTH,
        max_papers: int = MAX_PAPERS,
        min_rel: float = MIN_RELEVANCE,
    ):
        self.max_depth = max_depth
        self.max_papers = max_papers
        self.min_rel = min_rel

        cache = Cache()
        self.arxiv = ArxivClient(cache)
        self.s2 = S2Client(cache)
        self.oa = OAClient(cache)

        self.visited: Set[str] = set()
        self.title_hashes: Set[str] = set()

    def _prepare_root(self, pdf_path: str) -> Paper:
        print(f"\n{'=' * 60}\n  Citation Tree Builder\n{'=' * 60}")
        print(f"\n  Extracting: {os.path.basename(pdf_path)}")
        info = extract_pdf(pdf_path)

        title = info["title"] or "Unknown Paper"
        arxiv_id = info["arxiv_id"]
        print(f"  Title : {title}")
        if arxiv_id:
            print(f"  arXiv : {arxiv_id}")
        print(f"  References: {len(info['references'])} extracted")

        root = self._lookup(title, arxiv_id)

        if root:
            pass  
        else:
            root = Paper(
                id=f"local:{hashlib.md5(title.encode()).hexdigest()[:12]}",
                title=title,
                abstract=info["abstract"],
                full_text=info["text"],
                arxiv_id=arxiv_id,
                source="local",
            )
        # Always apply the locally-extracted PDF text to the root paper, even when
        # _lookup succeeded and returned an API paper (which has no full_text).
        if not root.full_text and info["text"]:
            root.full_text = info["text"]
        root.depth = 0
        return root, info
    
    def _postprocess(self, tree: CitationTree, is_reference: bool) -> None:
        print("\n  Downloading PDFs and extracting full text")
        for paper in tree.papers.values():
            if paper.full_text:
                continue  
            local_path = download_pdf(paper, PDFS_DIR, rate_limit=RATE_LIMIT)
            if local_path:
                extracted = extract_pdf(local_path)
                paper.full_text = extracted.get("text") or ""
                if not paper.abstract and extracted.get("abstract"):
                    paper.abstract = extracted["abstract"]
                status = f"{len(paper.full_text)} chars"
            else:
                status = "no PDF"
            print(f"    {paper.title[:50]}… [{status}]")

        print("\n  Generating improvement explanations")
        for paper in tree.papers.values():
            if paper.parent_id and paper.parent_id in tree.papers:
                parent = tree.papers[paper.parent_id]
                paper.similarity_to_parent = compute_similarity(parent.abstract or parent.title, paper.abstract or paper.title,)
                paper.improvement = generate_improvement_explanation(parent, paper, is_reference)

    # Builds the citation tree from a given pdf by extracting information for the root of the tree
    # and then expanding the tree iteratively by looking up references and related papers via APIs, 
    # scoring them for relevance, and adding the most relevant ones as child nodes.
    def build_reference_tree(self, pdf_path: str) -> CitationTree:
        root, info = self._prepare_root(pdf_path)
        tree = CitationTree(root=root)
        tree.papers[root.id] = root
        self.visited.add(root.id)
        self.title_hashes.add(title_hash(root.title))
        self._expand_references(tree, root, info["references"])
        self._postprocess(tree, is_reference=True)
        return tree
    
    def build_citation_tree(self, pdf_path: str) -> CitationTree:
        root, _ = self._prepare_root(pdf_path)
        tree = CitationTree(root=root)
        tree.papers[root.id] = root
        self.visited.add(root.id)
        self.title_hashes.add(title_hash(root.title))
        self._expand_citations(tree, root)
        self._postprocess(tree, is_reference=False)
        return tree
    
    # Looks up a paper by title and/or arXiv ID across multiple APIs, and returns a Paper object if found
    def _lookup(self, title: str, arxiv_id: str | None = None) -> Paper | None:
        if arxiv_id:
            p = self.s2.get_by_arxiv(arxiv_id)
            if p:
                return p
            p = self.arxiv.get_by_id(arxiv_id)
            if p:
                return p
        if title:
            for client in (self.s2, self.oa):
                for p in client.search(title, limit=3):
                    if titles_match(title, p.title):
                        return p
        return None

    # Recursively expands the tree by looking up references and related papers, scoring them, 
    # and adding the most relevant ones as child nodes.
    def _expand_references(self, tree: CitationTree, paper: Paper, local_refs: list | None = None,) -> None:
        if paper.depth >= self.max_depth or len(tree.papers) >= self.max_papers:
            return

        indent = "  " * (paper.depth + 1)
        print(f"{indent}[depth {paper.depth}] {paper.title[:50]}…")

        related: List[Paper] = []

        # checking multiple APIs for references, starting with the most specific (S2 and OA with ID) 
        # and falling back to title search in other sources if needed
        client_prefix_list = [("s2:", "S2", self.s2), ("oa:", "OA", self.oa)]
        for prefix, label, client in client_prefix_list:
            if paper.id.startswith(prefix):
                api_id = paper.id[len(prefix):]
                refs = client.get_references(api_id, limit=20)
                print(f"{indent}  {label}: {len(refs)} references")
                related.extend(refs)

        if not related:
            for prefix, label, client in client_prefix_list:
                if paper.id.startswith(prefix) and paper.title:
                    for found in client.search(paper.title, limit=3):
                        if titles_match(paper.title, found.title):
                            api_id = found.id.split(":", 1)[1]
                            refs = client.get_references(api_id, limit=20)
                            print(f"{indent}  {label} (from other sources): {len(refs)} references")
                            related.extend(refs)
                            break
                    if related:
                        break
        
        # If no references found via APIs, try arXiv search by title keywords (for arXiv papers)
        if paper.id.startswith(("arxiv:", "local:")) and not related:
            if paper.arxiv_id:
                s2_paper = self.s2.get_by_arxiv(paper.arxiv_id)
                if s2_paper:
                    s2_id = s2_paper.id[3:]
                    refs = self.s2.get_references(s2_id, limit=20)
                    print(f"{indent}  S2 (via arXiv): {len(refs)} refs")
                    related.extend(refs)
            if not related and paper.title:
                for label, client in (("S2", self.s2), ("OA", self.oa)):
                    for found in client.search(paper.title, limit=3):
                        if titles_match(paper.title, found.title):
                            api_id = found.id.split(":", 1)[1]
                            refs = client.get_references(api_id, limit=20)
                            print(f"{indent}  {label} (via search): {len(refs)} refs")
                            related.extend(refs)
                            break
                    if related:
                        break

        # If still no references, try a keyword search on arXiv (for local papers with no arXiv ID but a title)
        if paper.title:
            kw = " ".join(list(important_words(paper.title)))
            if kw:
                ax = self.arxiv.search(kw, max_results=10)
                print(f"{indent}  arXiv search: {len(ax)}")
                for p in ax:
                    p.relation_type = "related"
                related.extend(ax)

        if len(related) < 5 and local_refs:
            for ref in local_refs[:10]:
                if ref.get("title"):
                    found = self._lookup(ref["title"])
                    if found:
                        found.relation_type = "reference"
                        related.append(found)

        # Score and filter candidates based on relevance to the current paper, 
        # and add the most relevant ones as children in the tree
        scored = self._score(paper, related)
        filtered: list[tuple[Paper, float]] = []

        for p, sc in scored:
            if sc < self.min_rel:
                continue

            if self._is_same_paper(paper, p) or self._is_same_paper(tree.root, p):
                continue

            th = title_hash(p.title)
            if p.id in self.visited or th in self.title_hashes:
                continue

            sim = compute_similarity(paper.abstract or paper.title, p.abstract or p.title)

            if sim < self.min_rel:
                continue

            p.similarity_to_parent = sim
            filtered.append((p, sc))
            self.visited.add(p.id)
            self.title_hashes.add(th)

        filtered.sort(key=lambda x: -x[1])
        to_add = filtered[: min(8, self.max_papers - len(tree.papers))]

        if not to_add and related:
            best = max((sc for _, sc in scored), default=0)
            print(
                f"{indent}  {len(related)} candidates scored, "
                f"best={best:.3f}, threshold={self.min_rel} — none passed"
            )

        for child, sc in to_add:
            child.depth = paper.depth + 1
            child.parent_id = paper.id
            child.relevance_score = sc
            tree.papers[child.id] = child
            tree.edges.append((paper.id, child.id, child.relation_type))

        for child, _ in to_add:
            if len(tree.papers) < self.max_papers:
                self._expand_references(tree, child)

    # Return True if *a* and *b* are the same paper (self-citation guard)
    @staticmethod
    def _is_same_paper(a: Paper, b: Paper) -> bool:
        
        if a.id == b.id:
            return True
       
        if a.arxiv_id and b.arxiv_id and a.arxiv_id == b.arxiv_id:
            return True
        
        if a.doi and b.doi and a.doi.lower() == b.doi.lower():
            return True
        
        wa, wb = important_words(a.title), important_words(b.title)
        if wa and wb:
            overlap = len(wa & wb) / max(1, min(len(wa), len(wb)))
            if overlap > 0.85:
                return True
        return False
   
    # Scores candidate papers for relevance to the source suing overlap coefficient (intersection / min set size),
    # with a bonus for title word overlap, category overlap, and citation count, and for direct references/citations
    def _score(self, source: Paper, candidates: List[Paper]) -> List[Tuple[Paper, float]]:
        src_words = important_words(source.title + " " + (source.abstract or ""))
       
        src_title_words = important_words(source.title)
        src_cats = set(source.categories)
        results: list[tuple[Paper, float]] = []
        for p in candidates:
            if not p or not p.title:
                continue

            sc = 0.0
            pw = important_words(p.title + " " + (p.abstract or ""))
            p_title_words = important_words(p.title)
            if pw and src_words:
                overlap = len(src_words & pw) / max(1, min(len(src_words), len(pw)))
                sc += overlap * 0.35
                
                if src_title_words and p_title_words:
                    title_overlap = len(src_title_words & p_title_words) / max(
                        1, min(len(src_title_words), len(p_title_words))
                    )
                    sc += title_overlap * 0.15
            if p.categories and src_cats:
                sc += (len(src_cats & set(p.categories))/ max(1, len(src_cats))* 0.2)
            if p.citations_count:
                sc += min(p.citations_count / 500, 0.15)
            if p.relation_type in ("reference", "citation"):
                sc += 0.15
            elif p.relation_type == "related":
                sc += 0.05
            results.append((p, sc))
            
        return results
    
    def _get_citations(self, paper: Paper, limit: int = 20) -> List[Paper]:
        citations: List[Paper] = []

        s2_id = None

        if paper.id.startswith("s2:"):
            s2_id = paper.id.split(":", 1)[1]

        elif paper.arxiv_id:
            s2_paper = self.s2.get_by_arxiv(paper.arxiv_id)
            if s2_paper:
                s2_id = s2_paper.id.split(":", 1)[1]

        if not s2_id and paper.title:
            for p in self.s2.search(paper.title, limit=3):
                if titles_match(paper.title, p.title):
                    s2_id = p.id.split(":", 1)[1]
                    break
        if s2_id:
            try:
                headers = {
                    "User-Agent": "CitationTree/2.0"
                }
                r = requests.get(
                    f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}/citations",
                    params={
                        "limit": limit,
                        "fields": "title,abstract,externalIds"
                    },
                    headers=headers,   
                    timeout=20,
                )
                if r.status_code == 200:
                    for item in r.json().get("data", []) or []:
                        cited = item.get("citingPaper")
                        if cited:
                            temp = Paper(
                                id=f"s2:{cited.get('paperId')}",
                                title=cited.get("title"),
                                abstract=cited.get("abstract"),
                                arxiv_id=(cited.get("externalIds") or {}).get("ArXiv"),
                                relation_type="citation",
                            )

                            enriched = self._lookup(temp.title, temp.arxiv_id)

                            if enriched:
                                enriched.relation_type = "citation"
                                citations.append(enriched)
                            else:
                                citations.append(temp)
            except Exception as e:
                print(f"  ⚠ citation fetch error: {e}")
        return citations


    # Recursively expands the tree by looking up citations, scoring them, 
    # and adding the most relevant ones as child nodes.
    def _expand_citations(self, tree: CitationTree, paper: Paper) -> None:
        if paper.depth >= self.max_depth or len(tree.papers) >= self.max_papers:
            return

        indent = "  " * (paper.depth + 1)
        print(f"{indent}[depth {paper.depth}] {paper.title[:50]}…")

        related = self._get_citations(paper)
        print(f"{indent}  Found {len(related)} citations")

        to_add = []

        for p in related:
            if not p or not p.title:
                continue

            if self._is_same_paper(paper, p) or self._is_same_paper(tree.root, p):
                continue

            th = title_hash(p.title)
            if p.id in self.visited or th in self.title_hashes:
                continue

            p.depth = paper.depth + 1
            p.parent_id = paper.id
            p.relevance_score = 1.0  
            p.similarity_to_parent = compute_similarity(
                paper.abstract or paper.title,
                p.abstract or p.title
            )

            to_add.append(p)
            self.visited.add(p.id)
            self.title_hashes.add(th)

        to_add = to_add[:8]

        for child in to_add:
            tree.papers[child.id] = child
            tree.edges.append((paper.id, child.id, "citation"))
            self._expand_citations(tree, child)