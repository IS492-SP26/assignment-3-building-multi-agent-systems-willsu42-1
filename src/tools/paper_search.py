"""
Paper Search Tool
Integrates with Semantic Scholar API for academic paper search.

This tool provides academic paper search functionality using the
Semantic Scholar API, which offers free access to a large corpus
of academic papers.
"""

from typing import List, Dict, Any, Optional
import os
import logging
import asyncio


class PaperSearchTool:
    """
    Tool for searching academic papers via Semantic Scholar API.
    
    Semantic Scholar provides free access to academic papers with
    rich metadata including citations, abstracts, and author information.
    API key is optional but recommended for higher rate limits.
    """

    def __init__(self, max_results: int = 10):
        """
        Initialize paper search tool.

        Args:
            max_results: Maximum number of papers to return
        """
        self.max_results = max_results
        self.logger = logging.getLogger("tools.paper_search")

        # API key is optional for Semantic Scholar
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        
        if not self.api_key:
            self.logger.info("No Semantic Scholar API key found. Using anonymous access (lower rate limits)")

    async def search(
        self,
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        min_citations: int = 0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for academic papers.

        Args:
            query: Search query
            year_from: Filter papers from this year onwards
            year_to: Filter papers up to this year
            min_citations: Minimum citation count
            **kwargs: Additional search parameters
                - fields: List of fields to retrieve

        Returns:
            List of papers with metadata format:
            {
                "paper_id": str,
                "title": str,
                "authors": List[{"name": str}],
                "year": int,
                "abstract": str,
                "citation_count": int,
                "url": str,
                "venue": str,
                "pdf_url": Optional[str],
            }
        """
        self.logger.info(f"Searching papers: {query}")

        # NOTE: Semantic Scholar now requires an API key even for anonymous access (HTTP 403).
        # Falling back to OpenAlex (https://openalex.org) — completely free, no key required.
        #
        # Semantic Scholar direct API (kept for reference, requires SEMANTIC_SCHOLAR_API_KEY):
        # api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        # headers = {"x-api-key": self.api_key} if self.api_key else {}
        #
        # Original Python library approach (kept for reference):
        # from semanticscholar import SemanticScholar
        # sch = SemanticScholar(api_key=self.api_key)
        # results = sch.search_paper(query, limit=self.max_results, ...)

        try:
            import aiohttp

            params = {
                "search": query,
                "per_page": self.max_results,
                "select": "id,title,authorships,publication_year,abstract_inverted_index,cited_by_count,primary_location,doi",
            }
            if year_from:
                params["filter"] = f"publication_year:>{year_from - 1}"

            # OpenAlex asks for a contact email in User-Agent for the polite pool
            headers = {"User-Agent": "research-assistant/1.0 (mailto:user@example.com)"}

            api_url = "https://api.openalex.org/works"
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params, headers=headers) as resp:
                    if resp.status != 200:
                        self.logger.error(f"OpenAlex API error: HTTP {resp.status}")
                        return []
                    data = await resp.json()

            papers = []
            for work in data.get("results", []):
                # Reconstruct abstract from inverted index
                abstract = ""
                inv_index = work.get("abstract_inverted_index") or {}
                if inv_index:
                    words = [""] * (max(max(v) for v in inv_index.values()) + 1)
                    for word, positions in inv_index.items():
                        for pos in positions:
                            words[pos] = word
                    abstract = " ".join(words)

                # Extract authors
                authors = [
                    {"name": a.get("author", {}).get("display_name", "")}
                    for a in work.get("authorships", [])[:10]
                ]

                # Extract venue
                primary = work.get("primary_location") or {}
                source = primary.get("source") or {}
                venue = source.get("display_name", "")

                # Build URL from DOI or OpenAlex ID
                doi = work.get("doi", "")
                url = doi if doi else work.get("id", "")

                papers.append({
                    "paper_id": work.get("id", ""),
                    "title": work.get("title", "Unknown"),
                    "authors": authors,
                    "year": work.get("publication_year"),
                    "abstract": abstract,
                    "citation_count": work.get("cited_by_count", 0),
                    "url": url,
                    "venue": venue,
                    "pdf_url": primary.get("pdf_url"),
                })

            papers = self._filter_by_year(papers, year_from, year_to)
            papers = self._filter_by_citations(papers, min_citations)
            self.logger.info(f"Found {len(papers)} papers via OpenAlex")
            return papers

        except Exception as e:
            self.logger.error(f"Error searching papers: {e}")
            return []

    async def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific paper.

        Args:
            paper_id: Semantic Scholar paper ID

        Returns:
            Detailed paper information
        """
        try:
            from semanticscholar import SemanticScholar
            
            sch = SemanticScholar(api_key=self.api_key)
            paper = sch.get_paper(paper_id)
            
            return {
                "paper_id": paper.paperId,
                "title": paper.title,
                "authors": [{"name": a.name} for a in paper.authors] if paper.authors else [],
                "year": paper.year,
                "abstract": paper.abstract,
                "citation_count": paper.citationCount,
                "url": paper.url,
                "venue": paper.venue,
                "pdf_url": paper.openAccessPdf.get("url") if paper.openAccessPdf else None,
            }
        except Exception as e:
            self.logger.error(f"Error getting paper details: {e}")
            return {}

    async def get_citations(self, paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get papers that cite this paper.

        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of citations to retrieve

        Returns:
            List of citing papers
        """
        try:
            from semanticscholar import SemanticScholar
            
            sch = SemanticScholar(api_key=self.api_key)
            paper = sch.get_paper(paper_id)
            citations = paper.citations[:limit] if paper.citations else []
            
            return [
                {
                    "paper_id": c.paperId,
                    "title": c.title,
                    "year": c.year,
                }
                for c in citations
            ]
        except Exception as e:
            self.logger.error(f"Error getting citations: {e}")
            return []

    async def get_references(self, paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get papers referenced by this paper.

        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of references to retrieve

        Returns:
            List of referenced papers
        """
        try:
            from semanticscholar import SemanticScholar
            
            sch = SemanticScholar(api_key=self.api_key)
            paper = sch.get_paper(paper_id)
            references = paper.references[:limit] if paper.references else []
            
            return [
                {
                    "paper_id": r.paperId,
                    "title": r.title,
                    "year": r.year,
                }
                for r in references
            ]
        except Exception as e:
            self.logger.error(f"Error getting references: {e}")
            return []

    def _parse_results(
        self,
        results: Any,
        year_from: Optional[int],
        year_to: Optional[int],
        min_citations: int
    ) -> List[Dict[str, Any]]:
        """
        Parse and filter search results from Semantic Scholar.
        
        Args:
            results: Raw results from Semantic Scholar API
            year_from: Minimum year filter
            year_to: Maximum year filter
            min_citations: Minimum citation count filter
            
        Returns:
            Filtered and formatted list of papers
        """
        papers = []
        
        for paper in results:
            # Skip papers without basic metadata
            if not paper or not hasattr(paper, 'title'):
                continue
                
            paper_dict = {
                "paper_id": paper.paperId if hasattr(paper, 'paperId') else None,
                "title": paper.title if hasattr(paper, 'title') else "Unknown",
                "authors": [{"name": a.name} for a in paper.authors] if hasattr(paper, 'authors') and paper.authors else [],
                "year": paper.year if hasattr(paper, 'year') else None,
                "abstract": paper.abstract if hasattr(paper, 'abstract') else "",
                "citation_count": paper.citationCount if hasattr(paper, 'citationCount') else 0,
                "url": paper.url if hasattr(paper, 'url') else "",
                "venue": paper.venue if hasattr(paper, 'venue') else "",
                "pdf_url": paper.openAccessPdf.get("url") if hasattr(paper, 'openAccessPdf') and paper.openAccessPdf else None,
            }
            
            papers.append(paper_dict)
        
        # Apply filters
        papers = self._filter_by_year(papers, year_from, year_to)
        papers = self._filter_by_citations(papers, min_citations)
        
        return papers

    def _filter_by_year(
        self,
        papers: List[Dict[str, Any]],
        year_from: Optional[int],
        year_to: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Filter papers by publication year."""
        filtered = papers
        if year_from:
            filtered = [p for p in filtered if p.get("year") and p.get("year") >= year_from]
        if year_to:
            filtered = [p for p in filtered if p.get("year") and p.get("year") <= year_to]
        return filtered

    def _filter_by_citations(
        self,
        papers: List[Dict[str, Any]],
        min_citations: int
    ) -> List[Dict[str, Any]]:
        """Filter papers by citation count."""
        return [p for p in papers if p.get("citation_count", 0) >= min_citations]


# Synchronous wrapper for use with AutoGen tools
def paper_search(query: str, max_results: int = 10, year_from: Optional[int] = None) -> str:
    """
    Synchronous wrapper for paper search (for AutoGen tool integration).

    Handles being called from both sync and async contexts. AutoGen executes
    tool functions inside an already-running event loop, so asyncio.run() would
    raise "This event loop is already running." We detect that case and fall back
    to running the coroutine in a fresh thread with its own loop.

    Args:
        query: Search query
        max_results: Maximum results to return
        year_from: Only return papers from this year onwards (e.g. 2019)

    Returns:
        Formatted string with paper metadata ready for the Researcher agent
    """
    import concurrent.futures

    tool = PaperSearchTool(max_results=max_results)

    try:
        asyncio.get_running_loop()
        # Already inside a running loop (e.g. AutoGen async context) —
        # run the coroutine in a background thread with its own loop
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            results = pool.submit(asyncio.run, tool.search(query, year_from=year_from)).result()
    except RuntimeError:
        # No running loop — safe to call asyncio.run() directly
        results = asyncio.run(tool.search(query, year_from=year_from))

    if not results:
        return f"No academic papers found for '{query}'."

    # Format results using [Source N] markers the Researcher agent is
    # instructed to use when reporting findings
    lines = [f"Academic paper search results for '{query}' ({len(results)} papers):\n"]
    for i, paper in enumerate(results, 1):
        # Build author string: first 3 authors, then et al.
        author_names = [a["name"] for a in paper["authors"][:3]]
        authors = ", ".join(author_names)
        if len(paper["authors"]) > 3:
            authors += " et al."

        venue = f" | {paper['venue']}" if paper.get("venue") else ""
        citations = paper.get("citation_count", 0)
        year = paper.get("year", "n.d.")

        lines.append(f"[Source {i}] {paper['title']}")
        lines.append(f"Authors: {authors or 'Unknown'} ({year}){venue} | Citations: {citations}")
        lines.append(f"URL: {paper.get('url', 'N/A')}")
        if paper.get("pdf_url"):
            lines.append(f"PDF: {paper['pdf_url']}")
        if paper.get("abstract"):
            abstract = paper["abstract"]
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            lines.append(f"Abstract: {abstract}")
        lines.append("")  # blank line between results

    return "\n".join(lines)
