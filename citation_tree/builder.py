"""Tree builder — orchestrates PDF extraction, API discovery, and ML scoring."""
# How tree creation works:
#
# 1. Start with the root PDF:
#    - We extract its title, abstract, full text, and references using extract_pdf().
#
# 2. Try to identify the root paper via APIs (_lookup):
#    - If found, we use the API version (has IDs, citations, etc.).
#    - If not, we create a local Paper object using extracted PDF info.
#
# 3. Build the tree structure:
#    - For each paper, we fetch related papers (references or citations) using APIs.
#    - At this stage, we use metadata (title, abstract, IDs) — no PDFs yet.
#
# 4. Score and filter candidates:
#    - We compute a cheap "relevance score" (keyword overlap, categories, etc.).
#    - We drop bad matches (low relevance, duplicates, wrong year, etc.).
#    - Then we compute semantic similarity (more expensive, more accurate).
#    - Only the best papers are kept as children.
#
# 5. Add children to the tree:
#    - Each selected paper becomes a node.
#    - We store parent-child relationships as edges.
#    - Then we recursively expand each child (until depth/size limits are hit).
#
# 6. Postprocess:
#    - Now that we have a smaller, relevant tree, we download PDFs only for those papers.
#    - We extract full text and better abstracts using extract_pdf().
#
# 7. Final enhancements:
#    - Compute semantic similarity between parent and child papers.
#    - Generate LLM-based explanations of how papers relate/improve on each other.
#
# Result:
#    - A tree of papers (nodes) connected by references/citations (edges),
#    - built efficiently using APIs first, then enriched with PDFs only for important papers.
# Note: semantic scholar arxiv and open alex are used for extracting references and citations, and arxiv is used for downloading the pdfs

from __future__ import annotations

import hashlib
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set, Tuple
from citation_tree.cache import GlobalRequestGate
from citation_tree.config import GLOBAL_ARXIV_MIN_INTERVAL
import time

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
    extract_abstract_with_llm,
    generate_improvement_explanation,
    is_similarity_available,
)
from citation_tree.models import CitationTree, Paper
from citation_tree.pdf import  extract_pdf, normalize_arxiv_id
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
    
    # creates the root of the tree
    def _prepare_root(self, pdf_path: str) -> Paper:
        print(f"\n{'=' * 60}\n  Citation Tree Builder\n{'=' * 60}")

        print(f"\n  Extracting: {os.path.basename(pdf_path)}")
        info = extract_pdf(pdf_path)
        title = info["title"] or "Unknown Paper"
        arxiv_id = info["arxiv_id"]

        normalized_arxiv = normalize_arxiv_id(arxiv_id)
        if normalized_arxiv:
            normalized_arxiv = normalized_arxiv.split("v", 1)[0]
        if normalized_arxiv:
            s2_paper = self.s2.get_by_arxiv(normalized_arxiv)
            if s2_paper and s2_paper.title:
                title = s2_paper.title
        print(f"  Title : {title}")
        if arxiv_id:
            print(f"  arXiv : {arxiv_id}")
        print(f"  References: {len(info['references'])} extracted")
        if not self.similarity_available:
            print("  Similarity model unavailable: semantic filter disabled")

      
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
        # _lookup succeeded and returned an API paper (which has no full_text)
        if not root.full_text and info["text"]:
            root.full_text = info["text"]
        root.depth = 0
        return root, info
    
    # after building the tree, downloads PDFs for all papers in the tree, extracts their text, 
    # and generates improvement explanations using the LLM
    def _postprocess(self, tree: CitationTree, is_reference: bool) -> None:
        print("\n  Downloading PDFs and extracting full text")
        papers_to_process = [p for p in tree.papers.values() if not p.full_text]
        
        # downloads and extracts the full text for a paper
        def _hydrate_full_text(paper: Paper) -> tuple[Paper, str]:
            local_path = self.download_pdf(paper, PDFS_DIR, rate_limit=RATE_LIMIT)
            if local_path:
                extracted = extract_pdf(local_path)
                paper.url = local_path
                paper.arxiv_id = extracted["arxiv_id"]
                paper.full_text = extracted.get("text") or ""
                if extracted.get("abstract"):
                    paper.abstract = extracted["abstract"]
                print(f"    Extracted full text for {paper.title[:50]} stored at {local_path}")
                print(f"  ArXiv ID: {paper.arxiv_id}\n\n")
                return paper, f"{len(paper.full_text)} chars"
            return paper, "no PDF"

        if papers_to_process:
            with ThreadPoolExecutor(max_workers=MAX_POSTPROCESS_WORKERS) as ex:
                futures = [ex.submit(_hydrate_full_text, p) for p in papers_to_process]
                for fut in as_completed(futures):
                    paper, status = fut.result()
                    print(f"    {paper.title[:50]}… [{status}]")

        print("\n  Backfilling missing abstracts")
        for paper in tree.papers.values():
            if (paper.abstract or "").strip():
                continue

            looked_up = self._lookup_with_abstract(paper.title, paper.arxiv_id)
            if looked_up and (looked_up.abstract or "").strip():
                paper.abstract = looked_up.abstract
                if not paper.arxiv_id and looked_up.arxiv_id:
                    paper.arxiv_id = looked_up.arxiv_id
                if not paper.pdf_url and looked_up.pdf_url:
                    paper.pdf_url = looked_up.pdf_url
                print(f"    {paper.title[:50]}… [metadata abstract]")
                continue

            if (paper.full_text or "").strip():
                inferred = extract_abstract_with_llm(paper.full_text, max_chunks=3)
                if inferred:
                    paper.abstract = inferred
                    print(f"    {paper.title[:50]}… [LLM abstract]")
                    continue

            # Keep abstracts non-empty when metadata/PDF extraction fails.
            paper.abstract = (
                f"Abstract unavailable from source metadata and PDF extraction for '{paper.title}'."
            )
            print(f"    {paper.title[:50]}… [fallback abstract]")

        print("\n  Generating improvement explanations")
        for paper in tree.papers.values():
            if paper.parent_id and paper.parent_id in tree.papers:
                parent = tree.papers[paper.parent_id]
                paper.similarity_to_parent = compute_similarity(parent.abstract or parent.title, paper.abstract or paper.title,)

                explanation = generate_improvement_explanation(parent, paper, is_reference)
                if explanation:
                    paper.improvement = explanation

        tree.root.improvement = tree.root.improvement or "Root paper of this tree (no parent paper)."
    
    # returns whether the paper should be considered (doi papers are normally paywalled or closed off)
    @staticmethod
    def _is_open_access_candidate(paper: Paper) -> bool:
        if paper.arxiv_id:
            return True
        if paper.is_open_access is True:
            return True
        if paper.pdf_url:
            return True
        if paper.doi:
            return False
        return True

    # Builds the reference tree from a given pdf by extracting information for the root of the tree
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
    
    # Builds the citation tree from a given pdf by extracting information for the root of the tree
    # and then expanding the tree iteratively by looking up citations and related papers via APIs, 
    # scoring them for relevance, and adding the most relevant ones as child nodes.
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
            normalized_arxiv = normalize_arxiv_id(arxiv_id)
            if normalized_arxiv:
                normalized_arxiv = normalized_arxiv.split("v", 1)[0]
                p = self.s2.get_by_arxiv(normalized_arxiv)
                if p:
                    return p

                p = self.arxiv.get_by_id(normalized_arxiv)
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

    # Returns the best match with a non-empty abstract when available.
    def _lookup_with_abstract(self, title: str, arxiv_id: str | None = None) -> Paper | None:
        direct = self._lookup(title, arxiv_id)
        if direct and (direct.abstract or "").strip():
            return direct

        if not title:
            return None

        query_variants: list[str] = []
        for q in (title, " ".join(title.split()[:10])):
            q = (q or "").strip()
            if q and q not in query_variants:
                query_variants.append(q)

        for query in query_variants:
            for client in (self.s2, self.oa, self.arxiv):
                if client is self.arxiv:
                    candidates = client.search(query, max_results=8)
                else:
                    candidates = client.search(query, limit=8)

                for candidate in candidates:
                    if not candidate or not candidate.title:
                        continue
                    if titles_match(title, candidate.title) and (candidate.abstract or "").strip():
                        return candidate

        return direct

    # Recursively expands the tree by looking up references and related papers, scoring them, 
    # and adding the most relevant ones as child nodes.
    # using score() for a less accurate, less intense keyword-based score to filter obvious irrelevant papers
    # and then applying expensive semantic similarity only on the remaining candidates for more accurate relevance
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
            "closed_access": 0,
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

        # checking APIs with title search if no references found via ID-based lookup
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
        
        # if this is an arXiv/local paper and no references were found,
        # try fetching references via Semantic Scholar using arXiv ID, or by
        # searching the paper title in S2/OpenAlex and retrieving references from matches
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
        # reference tree. This tree should represent actual references only
        if len(related) < 5 and local_refs:
            for ref in local_refs[:10]:
                if ref.get("title"):
                    found = self._lookup(ref["title"])
                    if found:
                        found.relation_type = "reference"
                        related.append(found)

        # If strict references are sparse, backfill with strong title-neighbor
        # papers so the configured branching/depth can still be explored.
        if len(related) < MAX_CHILDREN_PER_NODE and paper.title:
            seen_ids = {p.id for p in related if p and p.id}
            seen_titles = {title_hash(p.title) for p in related if p and p.title}
            target_pool = max(MAX_CHILDREN_PER_NODE * 4, 12)

            query_variants: list[str] = []
            for q in (paper.title, " ".join(paper.title.split()[:10])):
                q = (q or "").strip()
                if q and q not in query_variants:
                    query_variants.append(q)

            for client in (self.s2, self.oa, self.arxiv):
                for query in query_variants:
                    if client is self.arxiv:
                        found_candidates = client.search(query, max_results=target_pool)
                    else:
                        found_candidates = client.search(query, limit=target_pool)

                    for candidate in found_candidates:
                        if not candidate or not candidate.title:
                            continue

                        th = title_hash(candidate.title)
                        if candidate.id in seen_ids or th in seen_titles:
                            continue

                        candidate.relation_type = "reference"
                        related.append(candidate)
                        seen_ids.add(candidate.id)
                        seen_titles.add(th)

                        if len(related) >= target_pool:
                            break
                    if len(related) >= target_pool:
                        break
                if len(related) >= target_pool:
                    break

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

            if not self._is_open_access_candidate(p):
                drop_stats["closed_access"] += 1
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
                # Keep a stable signal for the renderer when semantic model is absent
                sim = min(1.0, sc)

            p.similarity_to_parent = sim
            filtered.append((p, sc, sim))
            self.visited.add(p.id)
            self.title_hashes.add(th)

        # If strict filtering leaves too few children, add best remaining
        # candidates with relaxed constraints to preserve branching.
        if len(filtered) < MAX_CHILDREN_PER_NODE:
            seen_ids = {p.id for p, _sc, _sim in filtered}
            seen_titles = {title_hash(p.title) for p, _sc, _sim in filtered}

            for p, sc in scored:
                if len(filtered) >= MAX_CHILDREN_PER_NODE:
                    break
                if not p or not p.title:
                    continue

                th = title_hash(p.title)
                if p.id in seen_ids or th in seen_titles:
                    continue
                if p.id in self.visited or th in self.title_hashes:
                    continue
                if self._is_same_paper(paper, p) or self._is_same_paper(tree.root, p):
                    continue

                sim = compute_similarity(paper.abstract or paper.title, p.abstract or p.title)
                if not self.similarity_available:
                    sim = min(1.0, max(sc, self.min_rel))

                p.similarity_to_parent = sim
                filtered.append((p, max(sc, self.min_rel), sim))
                self.visited.add(p.id)
                self.title_hashes.add(th)
                seen_ids.add(p.id)
                seen_titles.add(th)

        # Keep most semantically similar papers first - relevance score is tie-breaker.
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
                f"self={drop_stats['self_or_root']}, "
                f"closed={drop_stats['closed_access']})"
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

    # returns true if both papers are likely the same based on ID, title, and other metadata,
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
   
    # Scores candidate papers for relevance to the source using overlap coefficient (intersection / min set size),
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
    
    # retrieves the citations for a paper by looking up its ID and title across multiple APIs
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
        
        # remove duplicates across APIs while preserving order
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
            "closed_access": 0,
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

            if not self._is_open_access_candidate(p):
                drop_stats["closed_access"] += 1
                if DEBUG_PRINT_ALL_CITERS:
                    drop_details.append(
                        f"drop=closed_access id={p.id} doi={p.doi or 'unknown'} title={p.title}"
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

            # If semantic model is missing, use heuristic relevance as fallback ranking signal
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
        # citation pages (for example, many same-year fresh uploads), then break ties
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
                f"older={drop_stats['older_than_parent']}, "
                f"closed={drop_stats['closed_access']})"
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



    def _search_s2_pdf_by_title(self, title: str) -> str | None:
        try:
            for p in self.s2.search(title, limit=5):
                if titles_match(title, p.title):
                    if p.arxiv_id:
                        return f"https://arxiv.org/pdf/{p.arxiv_id}.pdf"
                    if p.pdf_url:
                        return p.pdf_url
        except Exception as e:
            print(f"  Semantic Scholar search failed: {e}")
        return None

    def _search_oa_pdf_by_title(self, title: str) -> str | None:
        try:
            for p in self.oa.search(title, limit=5):
                if titles_match(title, p.title):
                    if p.arxiv_id:
                        return f"https://arxiv.org/pdf/{p.arxiv_id}.pdf"
                    if p.pdf_url:
                        return p.pdf_url
        except Exception as e:
            print(f"  OpenAlex search failed: {e}")
        return None

    def _search_arxiv_pdf_by_title(self, title: str) -> str | None:
        try:
            query = title.replace(" ", "+")
            search_url = f"http://export.arxiv.org/api/query?search_query=ti:{query}&start=0&max_results=5"

            r = GlobalRequestGate.request(
                requests,
                "GET",
                search_url,
                group="arxiv",
                min_interval=GLOBAL_ARXIV_MIN_INTERVAL,
                timeout=20,
            )

            if r.status_code == 200:
                entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.DOTALL | re.I)

                for entry in entries:
                    id_match = re.search(r"<id>http://arxiv\.org/abs/(.*?)</id>", entry, re.I)
                    title_match = re.search(r"<title>(.*?)</title>", entry, re.DOTALL | re.I)

                    if not id_match or not title_match:
                        continue

                    candidate_id = id_match.group(1).strip()
                    candidate_title = " ".join(title_match.group(1).split()).strip()

                    if titles_match(title, candidate_title):
                        return f"https://arxiv.org/pdf/{candidate_id}.pdf"

        except Exception as e:
            print(f"  arXiv search failed: {e}")

        return None

    def _is_valid_pdf(self, path: str) -> bool:
        try:
            with open(path, "rb") as f:
                return f.read(5) == b"%PDF-"
        except Exception:
            return False
    
    # Downloads a paper's PDF to pdfs_dir if it is not already cached.
    # Uses arxiv_id as the filename (e.g. 1706.03762.pdf).
    # Returns the local path on success, or None if the paper has no arxiv_id or the download fails.
    def download_pdf(self, paper, pdfs_dir: str, rate_limit: float = 1.2) -> str | None:
        url = None

        latest_arxiv_id = getattr(paper, "arxiv_id", None)
        if latest_arxiv_id:
            paper.arxiv_id = latest_arxiv_id

        if latest_arxiv_id:
            url = f"https://arxiv.org/pdf/{latest_arxiv_id}.pdf"

        elif getattr(paper, "pdf_url", None):
            url = paper.pdf_url


        # using title to search arxiv api and get the first valid result
        elif getattr(paper, "title", None):
            searchers = [
                ("Semantic Scholar", self._search_s2_pdf_by_title),
                ("OpenAlex", self._search_oa_pdf_by_title),
                ("arXiv", self._search_arxiv_pdf_by_title),
            ]

            url = None
            for label, fn in searchers:
                url = fn(paper.title)
                if url:
                    print(f"  Found PDF via {label}: {url}")
                    break
            # try:
            #     query = paper.title.replace(" ", "+")
            #     search_url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=3"

            #     r = GlobalRequestGate.request(
            #         requests,
            #         "GET",
            #         search_url,
            #         group="arxiv",
            #         min_interval=GLOBAL_ARXIV_MIN_INTERVAL,
            #         timeout=20,
            #     )
                

            #     if r.status_code == 200:
            #         entries = re.findall(r"<id>http://arxiv.org/abs/(.*?)</id>", r.text)

            #         for arxiv_id in entries:
            #             if arxiv_id:
            #                 url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            #                 break

            # except Exception as e:
            #     print(f"  arXiv search failed: {e}")
        

        if not url:
            print(f"  No PDF available for: {paper.title[:60]}")
            return None

        if latest_arxiv_id:
            filename = f"{latest_arxiv_id}.pdf"
        else:
            safe_id = (paper.id or "unknown").replace(":", "_")
            filename = f"{safe_id}.pdf"

        local_path = os.path.join(pdfs_dir, filename)

        if os.path.exists(local_path):
            return local_path

        try:
            time.sleep(rate_limit)

            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; citation-tree-bot/1.0)"
            }

            resp = GlobalRequestGate.request(
                requests,
                "GET",
                url,
                group="pdf",
                min_interval=0.0,
                headers=headers,
                timeout=30,
                stream=True,
                allow_redirects=True,
            )

            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "").lower()

            if "pdf" not in content_type:
                print(f"  Not a PDF (content-type={content_type}): {url}")
                return None

            os.makedirs(pdfs_dir, exist_ok=True)

            with open(local_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
            
            if not self._is_valid_pdf(local_path):
                print(f"  Invalid PDF downloaded: {url}")
                os.remove(local_path)
                return None

            return local_path

        except Exception as exc:
            print(f"  PDF download failed ({paper.id}): {exc}")
            return None