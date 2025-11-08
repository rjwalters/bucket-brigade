#!/usr/bin/env python3
"""
Build Research Content for Website

This script collects research notebook entries, documentation, and experimental
results and formats them for display on the website.

Output:
  - web/public/research/notebook/ - Formatted notebook entries
  - web/public/research/library/ - Research documentation
  - web/public/research/index.json - Metadata for browser
"""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class ResearchContentBuilder:
    def __init__(self, root_dir: Path):
        self.root = root_dir
        self.notebook_src = root_dir / "research_notebook"
        self.library_src = root_dir / "experiments"
        self.docs_src = root_dir / "docs"
        self.output_dir = root_dir / "web" / "public" / "research"

    def parse_frontmatter(self, content: str) -> tuple[Dict[str, Any], str]:
        """Extract YAML frontmatter from markdown content."""
        frontmatter = {}
        body = content

        lines = content.split("\n")

        # Check for YAML frontmatter (--- ... ---)
        if lines[0].strip() == "---":
            # Find closing ---
            end_idx = None
            for i, line in enumerate(lines[1:], start=1):
                if line.strip() == "---":
                    end_idx = i
                    break

            if end_idx:
                # Parse YAML frontmatter
                yaml_lines = lines[1:end_idx]
                for line in yaml_lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')

                        # Parse arrays
                        if value.startswith("[") and value.endswith("]"):
                            # Simple array parsing
                            items = value[1:-1].split(",")
                            frontmatter[key] = [item.strip().strip('"\'') for item in items]
                        else:
                            frontmatter[key] = value

                # Body starts after closing ---
                body = "\n".join(lines[end_idx + 1:])
                return frontmatter, body

        # Fall back to old format if no YAML frontmatter
        if lines[0].startswith("# "):
            title = lines[0][2:].strip()
            frontmatter["title"] = title

            # Parse metadata from subsequent lines
            for i, line in enumerate(lines[1:10], start=1):  # Check first 10 lines
                if line.startswith("**Date**:"):
                    frontmatter["date"] = line.split(":", 1)[1].strip()
                elif line.startswith("**Author**:"):
                    frontmatter["author"] = line.split(":", 1)[1].strip()
                elif line.startswith("**Status**:"):
                    frontmatter["status"] = line.split(":", 1)[1].strip()
                elif line.startswith("**Tags**:"):
                    tags_str = line.split(":", 1)[1].strip()
                    # Extract tags from backticks: `tag1`, `tag2`
                    frontmatter["tags"] = re.findall(r"`([^`]+)`", tags_str)
                elif line.startswith("---") and i > 1:
                    body = "\n".join(lines[i + 1 :])
                    break

        return frontmatter, body

    def build_notebook_index(self) -> List[Dict[str, Any]]:
        """Build index of all notebook entries."""
        entries = []

        if not self.notebook_src.exists():
            print(f"Warning: Notebook directory not found: {self.notebook_src}")
            return entries

        for entry_file in sorted(self.notebook_src.glob("*.md"), reverse=True):
            content = entry_file.read_text()
            metadata, body = self.parse_frontmatter(content)

            # Extract date from filename if not in metadata
            filename = entry_file.stem
            date_match = re.match(r"(\d{4}-\d{2}-\d{2})", filename)
            if date_match and "date" not in metadata:
                metadata["date"] = date_match.group(1)

            # Create entry
            entry = {
                "id": filename,
                "filename": entry_file.name,
                "title": metadata.get("title", filename.replace("_", " ").title()),
                "date": metadata.get("date", "Unknown"),
                "author": metadata.get("author", "Research Team"),
                "status": metadata.get("status", ""),
                "tags": metadata.get("tags", []),
                "excerpt": self.extract_excerpt(body),
                "word_count": len(body.split()),
            }

            entries.append(entry)

        return entries

    def extract_excerpt(self, content: str, max_words: int = 50) -> str:
        """Extract first paragraph as excerpt."""
        # Remove markdown headers
        content = re.sub(r"^#+\s+.*$", "", content, flags=re.MULTILINE)
        # Remove markdown formatting
        content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)
        content = re.sub(r"`([^`]+)`", r"\1", content)
        # Get first paragraph
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs:
            return ""
        first_para = paragraphs[0]
        words = first_para.split()
        if len(words) > max_words:
            return " ".join(words[:max_words]) + "..."
        return first_para

    def build_library_index(self) -> List[Dict[str, Any]]:
        """Build index of research documentation."""
        library = []

        # Key documentation files
        doc_files = [
            # Experiments
            (
                self.library_src / "RESEARCH_SUMMARY.md",
                "research-summary",
                "Research Summary",
                "overview",
            ),
            (
                self.library_src / "RESEARCH_PROGRESS.md",
                "research-progress",
                "Research Progress",
                "progress",
            ),
            (
                self.library_src / "evolution" / "README.md",
                "evolution-readme",
                "Evolution Research",
                "evolution",
            ),
            (
                self.library_src / "nash" / "README.md",
                "nash-readme",
                "Nash Equilibrium Analysis",
                "nash",
            ),
            (
                self.library_src / "EVOLUTION_ANALYSIS_GUIDE.md",
                "evolution-guide",
                "Evolution Analysis Guide",
                "guide",
            ),
            # Docs
            (
                self.docs_src / "game_mechanics.md",
                "game-mechanics",
                "Game Mechanics",
                "design",
            ),
            (
                self.docs_src / "game-design" / "GAME_DYNAMICS.md",
                "game-dynamics",
                "Game Dynamics Specification",
                "design",
            ),
        ]

        for src_path, doc_id, title, category in doc_files:
            if not src_path.exists():
                print(f"Warning: Documentation file not found: {src_path}")
                continue

            content = src_path.read_text()
            metadata, body = self.parse_frontmatter(content)

            # Get file stats
            stat = src_path.stat()
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")

            library.append(
                {
                    "id": doc_id,
                    "title": title,
                    "filename": src_path.name,
                    "category": category,
                    "path": str(src_path.relative_to(self.root)),
                    "modified": modified,
                    "word_count": len(body.split()),
                    "excerpt": self.extract_excerpt(body),
                }
            )

        return library

    def copy_notebook_entries(self):
        """Copy notebook entries to website."""
        output_notebook = self.output_dir / "notebook"
        output_notebook.mkdir(parents=True, exist_ok=True)

        if not self.notebook_src.exists():
            print(f"Warning: Notebook directory not found: {self.notebook_src}")
            return

        for entry_file in self.notebook_src.glob("*.md"):
            dest = output_notebook / entry_file.name
            shutil.copy2(entry_file, dest)
            print(f"Copied: {entry_file.name} → notebook/")

    def copy_library_docs(self):
        """Copy library documentation to website."""
        output_library = self.output_dir / "library"
        output_library.mkdir(parents=True, exist_ok=True)

        # Key docs to copy
        docs_to_copy = [
            (self.library_src / "RESEARCH_SUMMARY.md", "research-summary.md"),
            (self.library_src / "RESEARCH_PROGRESS.md", "research-progress.md"),
            (
                self.library_src / "evolution" / "README.md",
                "evolution-research.md",
            ),
            (self.library_src / "nash" / "README.md", "nash-equilibrium.md"),
            (
                self.library_src / "EVOLUTION_ANALYSIS_GUIDE.md",
                "evolution-analysis-guide.md",
            ),
            (self.docs_src / "game_mechanics.md", "game-mechanics.md"),
            (
                self.docs_src / "game-design" / "GAME_DYNAMICS.md",
                "game-dynamics.md",
            ),
        ]

        for src, dest_name in docs_to_copy:
            if not src.exists():
                print(f"Warning: Source file not found: {src}")
                continue
            dest = output_library / dest_name
            shutil.copy2(src, dest)
            print(f"Copied: {src.name} → library/{dest_name}")

    def build_index(self):
        """Build complete index file."""
        index = {
            "generated": datetime.now().isoformat(),
            "notebook": {
                "title": "Research Notebook",
                "description": "Chronological record of research progress, insights, and experiments",
                "entries": self.build_notebook_index(),
            },
            "library": {
                "title": "Research Library",
                "description": "Documentation, guides, and comprehensive research reports",
                "documents": self.build_library_index(),
            },
            "tags": self.collect_all_tags(),
        }

        output_file = self.output_dir / "content_index.json"
        output_file.write_text(json.dumps(index, indent=2))
        print(f"\nGenerated: content_index.json")
        print(f"  - {len(index['notebook']['entries'])} notebook entries")
        print(f"  - {len(index['library']['documents'])} library documents")
        print(f"  - {len(index['tags'])} unique tags")

        return index

    def collect_all_tags(self) -> List[str]:
        """Collect all unique tags from notebook entries."""
        tags = set()
        notebook_entries = self.build_notebook_index()
        for entry in notebook_entries:
            tags.update(entry.get("tags", []))
        return sorted(list(tags))

    def build(self):
        """Build all research content."""
        print("Building research content for website...\n")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Copy content
        print("Copying notebook entries...")
        self.copy_notebook_entries()

        print("\nCopying library documentation...")
        self.copy_library_docs()

        # Build index
        print("\nBuilding content index...")
        self.build_index()

        print("\n✅ Research content build complete!")
        print(f"Output directory: {self.output_dir}")


def main():
    """Main entry point."""
    root = Path(__file__).parent.parent
    builder = ResearchContentBuilder(root)
    builder.build()


if __name__ == "__main__":
    main()
