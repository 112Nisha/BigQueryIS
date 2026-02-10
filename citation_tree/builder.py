"""Tree builder — orchestrates PDF extraction, API discovery, and ML scoring."""

from __future__ import annotations

import hashlib
import os
from typing import List, Set, Tuple

from citation_tree.cache import Cache
from citation_tree.clients import ArxivClient, OAClient, S2Client
from citation_tree.config import MAX_DEPTH, MAX_PAPERS, MIN_RELEVANCE
from citation_tree.ml import compute_similarity, generate_improvement_explanation
from citation_tree.models import CitationTree, Paper
from citation_tree.pdf import extract_pdf
from citation_tree.text_utils import important_words, title_hash, titles_match


class TreeBuilder:
    """Builds a CitationTree from a local PDF."""

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

    # ── public entry point ───────────────────────────────────────────────

    def build(self, pdf_path: str) -> CitationTree:
        print(f"\n{'=' * 60}\n  Citation Tree Builder\n{'=' * 60}")
        print(f"\n  Extracting: {os.path.basename(pdf_path)}")
        info = extract_pdf(pdf_path)

        title = info["title"] or "Unknown Paper"
        arxiv_id = info["arxiv_id"]
        print(f"  Title : {title[:70]}")
        if arxiv_id:
            print(f"  arXiv : {arxiv_id}")
        print(f"  Refs  : {len(info['references'])} extracted")

        root = self._lookup(title, arxiv_id) or Paper(
            id=f"local:{hashlib.md5(title.encode()).hexdigest()[:12]}",
            title=title,
            abstract=info["abstract"],
            arxiv_id=arxiv_id,
            source="local",
        )
        root.depth = 0

        tree = CitationTree(root=root)
        tree.papers[root.id] = root
        self.visited.add(root.id)
        self.title_hashes.add(title_hash(root.title))

        print(
            f"\n  Building tree (depth={self.max_depth}, max={self.max_papers})…"
        )
        self._expand(tree, root, info["references"])

        # ML pass — generate improvement explanations for every child
        print("\n  Generating improvement explanations…")
        for paper in tree.papers.values():
            if paper.parent_id and paper.parent_id in tree.papers:
                parent = tree.papers[paper.parent_id]
                paper.similarity_to_parent = compute_similarity(
                    parent.abstract or parent.title,
                    paper.abstract or paper.title,
                )
                paper.improvement = generate_improvement_explanation(
                    parent, paper
                )

        return tree

    # ── internal helpers ─────────────────────────────────────────────────

    def _lookup(
        self, title: str, arxiv_id: str | None = None
    ) -> Paper | None:
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

    def _expand(
        self,
        tree: CitationTree,
        paper: Paper,
        local_refs: list | None = None,
    ):
        if paper.depth >= self.max_depth or len(tree.papers) >= self.max_papers:
            return

        indent = "  " * (paper.depth + 1)
        print(f"{indent}[depth {paper.depth}] {paper.title[:50]}…")

        related: List[Paper] = []

        # Fetch refs/cites from whichever API sourced this paper
        for prefix, label, client in (
            ("s2:", "S2", self.s2),
            ("oa:", "OA", self.oa),
        ):
            if paper.id.startswith(prefix):
                api_id = paper.id[len(prefix):]
                refs = client.get_references(api_id, limit=20)
                cites = client.get_citations(api_id, limit=20)
                print(
                    f"{indent}  {label}: {len(refs)} refs, {len(cites)} cites"
                )
                related.extend(refs)
                related.extend(cites)

        # arXiv keyword search
        if paper.title:
            kw = " ".join(list(important_words(paper.title))[:6])
            if kw:
                ax = self.arxiv.search(kw, max_results=10)
                print(f"{indent}  arXiv search: {len(ax)}")
                for p in ax:
                    p.relation_type = "related"
                related.extend(ax)

        # Fall back to extracted references
        if len(related) < 5 and local_refs:
            for ref in local_refs[:10]:
                if ref.get("title"):
                    found = self._lookup(ref["title"])
                    if found:
                        found.relation_type = "reference"
                        related.append(found)

        # Score, de-duplicate, pick top-N
        scored = self._score(paper, related)
        filtered: list[tuple[Paper, float]] = []
        for p, sc in scored:
            th = title_hash(p.title)
            if (
                p.id not in self.visited
                and th not in self.title_hashes
                and sc >= self.min_rel
            ):
                filtered.append((p, sc))
                self.visited.add(p.id)
                self.title_hashes.add(th)
        filtered.sort(key=lambda x: -x[1])
        to_add = filtered[: min(8, self.max_papers - len(tree.papers))]

        for child, sc in to_add:
            child.depth = paper.depth + 1
            child.parent_id = paper.id
            child.relevance_score = sc
            tree.papers[child.id] = child
            tree.edges.append((paper.id, child.id, child.relation_type))

        for child, _ in to_add:
            if len(tree.papers) < self.max_papers:
                self._expand(tree, child)

    def _score(
        self, source: Paper, candidates: List[Paper]
    ) -> List[Tuple[Paper, float]]:
        src_words = important_words(
            source.title + " " + (source.abstract or "")
        )
        src_cats = set(source.categories)
        results: list[tuple[Paper, float]] = []
        for p in candidates:
            if not p or not p.title:
                continue
            sc = 0.0
            pw = important_words(p.title + " " + (p.abstract or ""))
            if pw and src_words:
                sc += len(src_words & pw) / max(1, len(src_words | pw)) * 0.4
            if p.categories and src_cats:
                sc += (
                    len(src_cats & set(p.categories))
                    / max(1, len(src_cats))
                    * 0.3
                )
            if p.citations_count:
                sc += min(p.citations_count / 500, 0.2)
            if p.relation_type in ("reference", "citation"):
                sc += 0.1
            results.append((p, sc))
        return results
