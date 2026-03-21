"""Tree builder — orchestrates PDF extraction, API discovery, and ML scoring."""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set, Tuple

from citation_tree.cache import Cache
from citation_tree.clients import ArxivClient, OAClient, S2Client
from citation_tree.config import (
    API_CITATION_LIMIT,
    API_REFERENCE_LIMIT,
    DEBUG_PRINT_ALL_CITERS,
    MAX_FETCH_WORKERS,
    MAX_CHILDREN_PER_NODE,
    MAX_DEPTH,
    MAX_PAPERS,
    MAX_POSTPROCESS_WORKERS,
    MIN_RELEVANCE,
    PDFS_DIR,
    RATE_LIMIT,
)
from citation_tree.ml import (
    compute_similarity,
    generate_improvement_explanation,
    is_similarity_available,
    llm_explanations_enabled,
)
from citation_tree.models import CitationTree, Paper
from citation_tree.pdf import download_pdf, ensure_latest_pdf_path, extract_pdf
from citation_tree.text_utils import important_words, title_hash, titles_match

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
        self.similarity_available = is_similarity_available()

        self.visited: Set[str] = set()
        self.title_hashes: Set[str] = set()

    def _prepare_root(self, pdf_path: str) -> Paper:
        print(f"\n{'=' * 60}\n  Citation Tree Builder\n{'=' * 60}")
        latest_pdf_path = ensure_latest_pdf_path(pdf_path, rate_limit=RATE_LIMIT)
        if latest_pdf_path != pdf_path:
            print(
                f"\n  Using latest arXiv version: "
                f"{os.path.basename(pdf_path)} -> {os.path.basename(latest_pdf_path)}"
            )
        pdf_path = latest_pdf_path

        print(f"\n  Extracting: {os.path.basename(pdf_path)}")
        info = extract_pdf(pdf_path)

        title = info["title"] or "Unknown Paper"
        arxiv_id = info["arxiv_id"]
        print(f"  Title : {title}")
        if arxiv_id:
            print(f"  arXiv : {arxiv_id}")
        print(f"  References: {len(info['references'])} extracted")
        if not self.similarity_available:
            print("  Similarity model unavailable: semantic filter disabled")

      
        root = self._lookup(title, arxiv_id)


        if root:
            if info["abstract"]:
                root.abstract = info["abstract"]
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
        papers_to_process = [p for p in tree.papers.values() if not p.full_text]

        def _hydrate_full_text(paper: Paper) -> tuple[Paper, str]:
            local_path = download_pdf(paper, PDFS_DIR, rate_limit=RATE_LIMIT)
            if local_path:
                extracted = extract_pdf(local_path)
                paper.full_text = extracted.get("text") or ""
                if extracted.get("abstract"):
                    paper.abstract = extracted["abstract"]
                return paper, f"{len(paper.full_text)} chars"
            return paper, "no PDF"

        if papers_to_process:
            with ThreadPoolExecutor(max_workers=MAX_POSTPROCESS_WORKERS) as ex:
                futures = [ex.submit(_hydrate_full_text, p) for p in papers_to_process]
                for fut in as_completed(futures):
                    paper, status = fut.result()
                    print(f"    {paper.title[:50]}… [{status}]")

        if not llm_explanations_enabled():
            print("\n  Skipping improvement explanations (LLM disabled or key missing)")
            return

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
        drop_stats = {
            "below_relevance": 0,
            "below_similarity": 0,
            "self_or_root": 0,
            "duplicate": 0,
            "invalid": 0,
        }

        # checking multiple APIs for references, starting with the most specific (S2 and OA with ID) 
        # and falling back to title search in other sources if needed
        client_prefix_list = [("s2:", "S2", self.s2), ("oa:", "OA", self.oa)]
        for prefix, label, client in client_prefix_list:
            if paper.id.startswith(prefix):
                api_id = paper.id[len(prefix):]
                refs = client.get_references(api_id, limit=API_REFERENCE_LIMIT)
                print(f"{indent}  {label}: {len(refs)} references")
                related.extend(refs)

        if not related:
            for prefix, label, client in client_prefix_list:
                if paper.id.startswith(prefix) and paper.title:
                    for found in client.search(paper.title, limit=3):
                        if titles_match(paper.title, found.title):
                            api_id = found.id.split(":", 1)[1]
                            refs = client.get_references(api_id, limit=API_REFERENCE_LIMIT)
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
                    refs = self.s2.get_references(s2_id, limit=API_REFERENCE_LIMIT)
                    print(f"{indent}  S2 (via arXiv): {len(refs)} refs")
                    related.extend(refs)
            if not related and paper.title:
                for label, client in (("S2", self.s2), ("OA", self.oa)):
                    for found in client.search(paper.title, limit=3):
                        if titles_match(paper.title, found.title):
                            api_id = found.id.split(":", 1)[1]
                            refs = client.get_references(api_id, limit=API_REFERENCE_LIMIT)
                            print(f"{indent}  {label} (via search): {len(refs)} refs")
                            related.extend(refs)
                            break
                    if related:
                        break

        # NOTE: We intentionally do not add generic "related" papers in the
        # reference tree. This tree should represent actual references only.

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
        filtered: list[tuple[Paper, float, float]] = []

        for p, sc in scored:
            if sc < self.min_rel:
                drop_stats["below_relevance"] += 1
                continue

            if self._is_same_paper(paper, p) or self._is_same_paper(tree.root, p):
                drop_stats["self_or_root"] += 1
                continue

            th = title_hash(p.title)
            if p.id in self.visited or th in self.title_hashes:
                drop_stats["duplicate"] += 1
                continue

            sim = compute_similarity(paper.abstract or paper.title, p.abstract or p.title)

            if self.similarity_available and sim < self.min_rel:
                drop_stats["below_similarity"] += 1
                continue
            if not self.similarity_available:
                # Keep a stable signal for the renderer when semantic model is absent.
                sim = min(1.0, sc)

            p.similarity_to_parent = sim
            filtered.append((p, sc, sim))
            self.visited.add(p.id)
            self.title_hashes.add(th)

        # Keep most semantically similar papers first; relevance score is tie-breaker.
        filtered.sort(key=lambda x: (x[2], x[1]), reverse=True)
        child_cap = min(MAX_CHILDREN_PER_NODE, self.max_papers - len(tree.papers))
        to_add = filtered[:child_cap]

        if not to_add and related:
            best = max((sc for _, sc in scored), default=0)
            print(
                f"{indent}  {len(related)} candidates scored, "
                f"best={best:.3f}, threshold={self.min_rel} — none passed"
            )
        elif related:
            print(
                f"{indent}  kept={len(to_add)}/{len(related)} "
                f"(drop rel={drop_stats['below_relevance']}, "
                f"sim={drop_stats['below_similarity']}, "
                f"dup={drop_stats['duplicate']}, "
                f"self={drop_stats['self_or_root']})"
            )

        for child, sc, _sim in to_add:
            child.depth = paper.depth + 1
            child.parent_id = paper.id
            child.relevance_score = sc
            tree.papers[child.id] = child
            tree.edges.append((paper.id, child.id, child.relation_type))

        for child, _sc, _sim in to_add:
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
    
    def _get_citations(self, paper: Paper, limit: int = API_CITATION_LIMIT) -> List[Paper]:
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
        oa_id = None
        if paper.id.startswith("oa:"):
            oa_id = paper.id.split(":", 1)[1]
        elif paper.title:
            for p in self.oa.search(paper.title, limit=3):
                if titles_match(paper.title, p.title):
                    oa_id = p.id.split(":", 1)[1]
                    break

        def _fetch_s2() -> list[Paper]:
            if not s2_id:
                return []
            return self.s2.get_citations(s2_id, limit=limit)

        def _fetch_oa() -> list[Paper]:
            if not oa_id:
                return []
            return self.oa.get_citations(oa_id, limit=limit)

        with ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as ex:
            s2_future = ex.submit(_fetch_s2)
            oa_future = ex.submit(_fetch_oa)
            citations.extend(s2_future.result())
            citations.extend(oa_future.result())

        dedup: list[Paper] = []
        seen_ids: set[str] = set()
        seen_titles: set[str] = set()
        for p in citations:
            if not p or not p.title:
                continue
            th = title_hash(p.title)
            if p.id in seen_ids or th in seen_titles:
                continue
            seen_ids.add(p.id)
            seen_titles.add(th)
            dedup.append(p)
        return dedup


    # Recursively expands the tree by looking up citations, scoring them, 
    # and adding the most relevant ones as child nodes.
    def _expand_citations(self, tree: CitationTree, paper: Paper) -> None:
        if paper.depth >= self.max_depth or len(tree.papers) >= self.max_papers:
            return

        indent = "  " * (paper.depth + 1)
        print(f"{indent}[depth {paper.depth}] {paper.title[:50]}…")

        related = self._get_citations(paper)
        print(f"{indent}  Found {len(related)} citations")

        if DEBUG_PRINT_ALL_CITERS and related:
            print(f"{indent}  All citing-paper candidates for parent:")
            print(f"{indent}    parent={paper.title} ({paper.year or 'unknown'})")
            for idx, cand in enumerate(related, start=1):
                print(
                    f"{indent}    [{idx}] year={cand.year or 'unknown'} "
                    f"src={cand.source} cites={cand.citations_count or 0} "
                    f"id={cand.id} title={cand.title}"
                )

        # Keep displayed citation count consistent with fetched citation neighbors.
        if related:
            paper.citations_count = max(paper.citations_count or 0, len(related))

        to_add: list[tuple[Paper, float, float]] = []
        drop_stats = {
            "below_relevance": 0,
            "below_similarity": 0,
            "self_or_root": 0,
            "duplicate": 0,
            "invalid": 0,
            "older_than_parent": 0,
        }
        drop_details: list[str] = []

        relevance_by_id = {
            p.id: sc for p, sc in self._score(paper, related)
        }

        for p in related:
            if not p or not p.title:
                drop_stats["invalid"] += 1
                if DEBUG_PRINT_ALL_CITERS:
                    drop_details.append("drop=invalid title")
                continue

            rel_score = relevance_by_id.get(p.id, 0.0)
            if rel_score < self.min_rel:
                drop_stats["below_relevance"] += 1
                if DEBUG_PRINT_ALL_CITERS:
                    drop_details.append(
                        f"drop=below_relevance id={p.id} rel={rel_score:.3f} title={p.title}"
                    )
                continue

            if self._is_same_paper(paper, p) or self._is_same_paper(tree.root, p):
                drop_stats["self_or_root"] += 1
                if DEBUG_PRINT_ALL_CITERS:
                    drop_details.append(
                        f"drop=self/root id={p.id} year={p.year or 'unknown'} title={p.title}"
                    )
                continue

            # Citing papers should not be older than the cited paper.
            if (
                paper.year is not None
                and p.year is not None
                and p.year < paper.year
            ):
                drop_stats["older_than_parent"] += 1
                if DEBUG_PRINT_ALL_CITERS:
                    drop_details.append(
                        f"drop=older_than_parent id={p.id} year={p.year} parent_year={paper.year} title={p.title}"
                    )
                continue

            th = title_hash(p.title)
            if p.id in self.visited or th in self.title_hashes:
                drop_stats["duplicate"] += 1
                if DEBUG_PRINT_ALL_CITERS:
                    drop_details.append(
                        f"drop=duplicate id={p.id} year={p.year or 'unknown'} title={p.title}"
                    )
                continue

            p.depth = paper.depth + 1
            p.parent_id = paper.id
            sim = compute_similarity(
                paper.abstract or paper.title,
                p.abstract or p.title
            )

            # If semantic model is missing, use heuristic relevance as fallback ranking signal.
            if not self.similarity_available:
                sim = rel_score

            if self.similarity_available and sim < self.min_rel:
                drop_stats["below_similarity"] += 1
                if DEBUG_PRINT_ALL_CITERS:
                    drop_details.append(
                        f"drop=below_similarity id={p.id} sim={sim:.3f} title={p.title}"
                    )
                continue

            p.relevance_score = rel_score
            p.similarity_to_parent = sim

            to_add.append((p, p.relevance_score, sim))
            self.visited.add(p.id)
            self.title_hashes.add(th)

            if DEBUG_PRINT_ALL_CITERS:
                print(
                    f"{indent}    keep-candidate year={p.year or 'unknown'} "
                    f"sim={sim:.3f} rel={p.relevance_score:.3f} "
                    f"cites={p.citations_count or 0} id={p.id} title={p.title}"
                )

        # Prefer oldest valid citing papers first to avoid recency-skewed
        # citation pages (e.g., many same-year fresh uploads), then break ties
        # by semantic similarity/relevance.
        candidate_pool = to_add
        candidate_pool.sort(
            key=lambda x: (
                x[0].year if x[0].year is not None else 9999,
                -x[2],
                -x[1],
            )
        )
        child_cap = min(MAX_CHILDREN_PER_NODE, self.max_papers - len(tree.papers))
        to_add = candidate_pool[:child_cap]
        if related:
            selected_years = [str(c.year) for c, _sc, _sim in to_add if c.year is not None]
            print(
                f"{indent}  kept={len(to_add)}/{len(related)} "
                f"(drop rel={drop_stats['below_relevance']}, "
                f"sim={drop_stats['below_similarity']}, "
                f"dup={drop_stats['duplicate']}, "
                f"self={drop_stats['self_or_root']}, "
                f"older={drop_stats['older_than_parent']})"
            )
            if selected_years:
                print(f"{indent}  selected citation years: {', '.join(selected_years)}")
            if DEBUG_PRINT_ALL_CITERS:
                print(f"{indent}  Final selected top-{len(to_add)} (ranked):")
                for rank, (c, sc, sim) in enumerate(to_add, start=1):
                    print(
                        f"{indent}    #{rank} year={c.year or 'unknown'} "
                        f"sim={sim:.3f} rel={sc:.3f} cites={c.citations_count or 0} "
                        f"id={c.id} title={c.title}"
                    )
                if drop_details:
                    print(f"{indent}  Dropped candidates ({len(drop_details)}):")
                    for line in drop_details:
                        print(f"{indent}    {line}")

        for child, _sc, _sim in to_add:
            tree.papers[child.id] = child
            tree.edges.append((paper.id, child.id, "citation"))
            self._expand_citations(tree, child)