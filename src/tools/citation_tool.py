"""
Citation Tool
Formats citations and manages citation lists.

This tool provides citation formatting in multiple styles (primarily APA)
and manages a bibliography for research outputs.
"""

from typing import Dict, Any, List
from datetime import datetime
import re


class CitationTool:
    """
    Tool for formatting and managing citations.
    
    Features:
    - APA style formatting (7th edition)
    - Citation tracking and deduplication
    - Bibliography generation
    - Support for papers, articles, and web sources
    """

    def __init__(self, style: str = "apa"):
        """
        Initialize citation tool.

        Args:
            style: Citation style ("apa", "mla", "chicago", etc.)
        """
        self.style = style
        self.citations: List[Dict[str, Any]] = []
        self.citation_counter = 0

    def format_citation(self, source: Dict[str, Any]) -> str:
        """
        Format a source as a citation.

        Args:
            source: Source information dictionary with keys:
                - type: "article", "paper", "webpage", or "book"
                - authors: List of author dicts with "name" key
                - year: Publication year
                - title: Source title
                - venue: Journal/conference name (for papers)
                - url: Web URL
                - doi: DOI identifier (for papers)
                - site_name: Website name (for webpages)

        Returns:
            Formatted citation string in the specified style (default: APA)
        """
        source_type = source.get("type", "article")

        if self.style == "apa":
            return self._format_apa(source, source_type)
        elif self.style == "mla":
            return self._format_mla(source, source_type)
        else:
            return self._format_apa(source, source_type)

    def _format_apa(self, source: Dict[str, Any], source_type: str) -> str:
        """
        Format citation in APA style (7th edition).
        
        Supports:
        - Academic papers/articles
        - Webpages
        - Generic sources
        
        Args:
            source: Source information dictionary
            source_type: Type of source ("article", "paper", "webpage", etc.)
            
        Returns:
            APA-formatted citation string
        """
        if source_type == "article" or source_type == "paper":
            # Journal article or academic paper
            authors = source.get("authors", [])
            year = source.get("year", "n.d.")
            title = source.get("title", "Untitled")
            venue = source.get("venue", "")

            # Format authors
            author_str = self._format_authors_apa(authors)

            # Basic APA format for article
            citation = f"{author_str} ({year}). {title}."
            if venue:
                citation += f" {venue}."

            # Add DOI or URL if available
            doi = source.get("doi")
            url = source.get("url")
            if doi:
                citation += f" https://doi.org/{doi}"
            elif url:
                citation += f" {url}"

            return citation

        elif source_type == "webpage":
            # Web page
            authors = source.get("authors", [])
            year = source.get("year", datetime.now().year)
            title = source.get("title", "Untitled")
            url = source.get("url", "")
            site_name = source.get("site_name", "")

            author_str = self._format_authors_apa(authors) if authors else site_name

            citation = f"{author_str} ({year}). {title}."
            if url:
                citation += f" {url}"

            return citation

        else:
            # Generic fallback
            return f"{source.get('title', 'Unknown')} ({source.get('year', 'n.d.')})"

    def _format_mla(self, source: Dict[str, Any], source_type: str) -> str:
        """
        Format citation in MLA style (9th edition).
        
        Args:
            source: Source information dictionary
            source_type: Type of source
            
        Returns:
            MLA-formatted citation string
        """
        if source_type == "article" or source_type == "paper":
            # Journal article or academic paper
            authors = source.get("authors", [])
            year = source.get("year", "n.d.")
            title = source.get("title", "Untitled")
            venue = source.get("venue", "")
            
            # Format authors for MLA (First Last, and Second Last)
            author_str = self._format_authors_mla(authors)
            
            # MLA format: Author(s). "Article Title." Journal Name, Year.
            citation = f'{author_str}. "{title}."'
            if venue:
                citation += f" {venue},"
            citation += f" {year}."
            
            # Add URL if available
            url = source.get("url")
            if url:
                citation += f" {url}."
            
            return citation
            
        elif source_type == "webpage":
            # Web page
            authors = source.get("authors", [])
            title = source.get("title", "Untitled")
            site_name = source.get("site_name", "")
            year = source.get("year", "n.d.")
            url = source.get("url", "")
            
            author_str = self._format_authors_mla(authors) if authors else site_name
            
            # MLA format for webpage
            citation = f'{author_str}. "{title}."'
            if site_name:
                citation += f" {site_name},"
            citation += f" {year}."
            if url:
                citation += f" {url}."
            
            return citation
            
        else:
            # Generic fallback
            return f'{source.get("title", "Unknown")}. {source.get("year", "n.d.")}.'
    
    def _format_authors_mla(self, authors: List[Dict[str, Any]]) -> str:
        """
        Format author list in MLA style.
        
        MLA format:
        - 1 author: Last, First
        - 2 authors: Last1, First1, and Last2, First2
        - 3+ authors: Last1, First1, et al.
        
        Args:
            authors: List of author dictionaries with "name" key
            
        Returns:
            MLA-formatted author string
        """
        if not authors:
            return "Unknown Author"
        
        if len(authors) == 1:
            name = authors[0].get("name", "Unknown")
            return self._format_single_author_mla(name)
        
        elif len(authors) == 2:
            name1 = self._format_single_author_mla(authors[0].get("name", "Unknown"))
            name2 = self._format_single_author_mla(authors[1].get("name", "Unknown"))
            return f"{name1}, and {name2}"
        
        else:
            # 3+ authors - use et al.
            first_author = self._format_single_author_mla(authors[0].get("name", "Unknown"))
            return f"{first_author}, et al."
    
    def _format_single_author_mla(self, name: str) -> str:
        """
        Format a single author name in MLA style (Last, First).
        
        Args:
            name: Author's full name
            
        Returns:
            MLA-formatted name (Last, First)
        """
        if not name or name == "Unknown":
            return "Unknown"
        
        # If already in Last, First format, return as is
        if ',' in name:
            return name
        
        # Split name into parts
        parts = name.strip().split()
        if len(parts) == 1:
            return parts[0]
        
        # Assume last part is surname, rest are given names
        surname = parts[-1]
        given_names = " ".join(parts[:-1])
        
        return f"{surname}, {given_names}"

    def _format_authors_apa(self, authors: List[Dict[str, Any]]) -> str:
        """
        Format author list in APA style.
        
        APA 7th edition:
        - 1-2 authors: List all
        - 3-20 authors: List all
        - 21+ authors: First 19, then ..., then last
        
        For simplicity, we use "et al." for 3+ authors
        """
        if not authors:
            return "Unknown Author"

        if len(authors) == 1:
            name = authors[0].get("name", "Unknown")
            return self._format_single_author(name)

        elif len(authors) == 2:
            name1 = self._format_single_author(authors[0].get("name", "Unknown"))
            name2 = self._format_single_author(authors[1].get("name", "Unknown"))
            return f"{name1}, & {name2}"

        else:
            # More than 2 authors - use et al. for brevity
            first_author = self._format_single_author(authors[0].get("name", "Unknown"))
            return f"{first_author}, et al."
    
    def _format_single_author(self, name: str) -> str:
        """
        Format a single author name in APA style (Last, F. M.)
        
        Handles various name formats and extracts last name and initials.
        """
        if not name or name == "Unknown":
            return "Unknown"
        
        # If already in Last, F. format, return as is
        if ',' in name:
            return name
        
        # Split name into parts
        parts = name.strip().split()
        if len(parts) == 1:
            return parts[0]
        
        # Assume last part is surname, rest are given names
        surname = parts[-1]
        given_names = parts[:-1]
        
        # Create initials from given names
        initials = ". ".join([n[0].upper() for n in given_names if n]) + "."
        
        return f"{surname}, {initials}"

    def add_citation(self, source: Dict[str, Any]) -> int:
        """
        Add a source to the citation list with deduplication.
        
        Checks if a source with the same title already exists to avoid duplicates.

        Args:
            source: Source information dictionary

        Returns:
            Citation number/index (1-based)
        """
        # Check if already exists (deduplication by title)
        for i, existing in enumerate(self.citations):
            if existing.get("title") == source.get("title"):
                return i + 1

        # Add new citation
        self.citations.append(source)
        self.citation_counter += 1
        return self.citation_counter

    def get_citation_number(self, source: Dict[str, Any]) -> int:
        """Get the citation number for a source."""
        for i, existing in enumerate(self.citations):
            if existing.get("title") == source.get("title"):
                return i + 1
        return 0

    def generate_bibliography(self) -> List[str]:
        """
        Generate formatted bibliography from all citations.
        
        Citations are formatted according to the selected style and sorted
        alphabetically by the first author's last name (APA/MLA standard).

        Returns:
            List of formatted citation strings, sorted alphabetically
        """
        bibliography = []
        for source in self.citations:
            citation = self.format_citation(source)
            bibliography.append(citation)

        # Sort alphabetically (standard for APA and MLA)
        bibliography.sort()

        return bibliography

    def format_inline(self, source: Dict[str, Any]) -> str:
        """
        Format a source as an inline citation for use inside running text.

        Produces the short [Author, Year] form the Writer uses, e.g.:
            [Smith et al., 2022]
            [Nielsen, 1994]
            [ACM CHI, 2023]   ← fallback when no authors

        Args:
            source: Source information dictionary (same schema as format_citation)

        Returns:
            Inline citation string, e.g. "[Smith et al., 2022]"
        """
        authors = source.get("authors", [])
        year = source.get("year", "n.d.")

        if authors:
            first = authors[0].get("name", "")
            parts = first.strip().split()
            last_name = parts[-1] if parts else first
            if len(authors) > 1:
                author_str = f"{last_name} et al."
            else:
                author_str = last_name
        else:
            # Fall back to site name or a shortened title
            author_str = source.get("site_name") or source.get("title", "Unknown")[:30]

        return f"[{author_str}, {year}]"

    def format_references_section(self) -> str:
        """
        Generate a markdown References section from all tracked citations.

        Returns a block ready to append to the Writer's report:

            ## References
            [1] Smith, J. (2022). Title. Venue. https://...
            [2] ...

        Returns:
            Markdown string with numbered reference list
        """
        if not self.citations:
            return "## References\n\nNo sources cited."

        lines = ["## References\n"]
        for i, source in enumerate(self.citations, 1):
            lines.append(f"[{i}] {self.format_citation(source)}")

        return "\n".join(lines)

    def parse_sources_from_research(self, research_text: str) -> List[Dict[str, Any]]:
        """
        Parse [Source N] blocks written by the Researcher agent into structured
        citation dictionaries that can be tracked and formatted.

        Expects lines in the form produced by web_search / paper_search:
            [Source 1] Title — Author (Year)
            URL: https://...
            Authors: First Last (Year) | Venue | Citations: N
            Abstract: ...

        Args:
            research_text: Raw text output from the Researcher agent

        Returns:
            List of source dicts (title, url, authors, year, venue, type)
        """
        sources = []
        # Split on [Source N] markers
        blocks = re.split(r'\[Source\s+\d+\]', research_text)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            source: Dict[str, Any] = {"type": "article", "authors": []}

            lines = block.splitlines()

            # First non-empty line is the title (possibly "Title — Author (Year)")
            if lines:
                first = lines[0].strip()
                # Strip em-dash author suffix if present: "Title — Author (Year)"
                title_part = re.split(r'\s+[—–-]{1,2}\s+', first)[0].strip()
                source["title"] = title_part

            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue

                lower = line.lower()

                if lower.startswith("url:"):
                    url = line[4:].strip()
                    source["url"] = url
                    # Treat as webpage if not a known academic domain
                    if not any(d in url for d in ["semanticscholar", "arxiv", "doi.org", "acm.org", "ieee.org"]):
                        source["type"] = "webpage"

                elif lower.startswith("pdf:"):
                    source["pdf_url"] = line[4:].strip()

                elif lower.startswith("authors:"):
                    # "Authors: First Last, Second Name (2022) | Venue | Citations: N"
                    authors_raw = line[8:].strip()
                    # Extract year from parentheses
                    year_match = re.search(r'\((\d{4})\)', authors_raw)
                    if year_match:
                        source["year"] = int(year_match.group(1))
                    # Extract venue (between | separators)
                    parts = authors_raw.split("|")
                    if len(parts) >= 2:
                        venue_part = parts[1].strip()
                        if not venue_part.lower().startswith("citations"):
                            source["venue"] = venue_part
                    # Parse author names (before the year parenthesis)
                    names_part = re.split(r'\s*\(', authors_raw)[0]
                    for name in re.split(r',\s*(?:and\s+)?', names_part):
                        name = name.replace("et al.", "").strip()
                        if name:
                            source["authors"].append({"name": name})

                elif lower.startswith("published:"):
                    raw = line[10:].strip()
                    year_match = re.search(r'(\d{4})', raw)
                    if year_match and "year" not in source:
                        source["year"] = int(year_match.group(1))

                elif lower.startswith("abstract:") or lower.startswith("summary:"):
                    source["abstract"] = line.split(":", 1)[1].strip()

            if source.get("title"):
                sources.append(source)

        return sources

    def clear_citations(self):
        """Clear all citations."""
        self.citations = []
        self.citation_counter = 0


# Standalone function for AutoGen tool integration
def format_citations(research_text: str, style: str = "apa") -> str:
    """
    Parse sources from Researcher output and return a formatted References section.

    This is the AutoGen-callable wrapper. Pass in the Researcher's full text
    output; it will extract all [Source N] blocks, deduplicate them, and return
    a numbered References section ready to append to the Writer's report.

    Args:
        research_text: Raw text from the Researcher agent containing [Source N] blocks
        style: Citation style — "apa" (default) or "mla"

    Returns:
        Formatted References section as a markdown string
    """
    tool = CitationTool(style=style)
    sources = tool.parse_sources_from_research(research_text)

    if not sources:
        return "No sources could be parsed from the research text."

    for source in sources:
        tool.add_citation(source)

    return tool.format_references_section()
