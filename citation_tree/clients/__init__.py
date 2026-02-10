"""API clients for paper discovery (arXiv, Semantic Scholar, OpenAlex)."""

from citation_tree.clients.base import BaseClient
from citation_tree.clients.arxiv import ArxivClient
from citation_tree.clients.semantic_scholar import S2Client
from citation_tree.clients.openalex import OAClient

__all__ = ["BaseClient", "ArxivClient", "S2Client", "OAClient"]
