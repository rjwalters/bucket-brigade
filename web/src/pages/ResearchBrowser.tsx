import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface NotebookEntry {
	id: string;
	filename: string;
	title: string;
	date: string;
	author: string;
	status: string;
	tags: string[];
	excerpt: string;
	word_count: number;
}

interface LibraryDocument {
	id: string;
	title: string;
	filename: string;
	category: string;
	path: string;
	modified: string;
	word_count: number;
	excerpt: string;
}

interface ContentIndex {
	generated: string;
	notebook: {
		title: string;
		description: string;
		entries: NotebookEntry[];
	};
	library: {
		title: string;
		description: string;
		documents: LibraryDocument[];
	};
	tags: string[];
}

type ViewMode = "notebook" | "library";

export default function ResearchBrowser() {
	const [index, setIndex] = useState<ContentIndex | null>(null);
	const [selectedEntry, setSelectedEntry] = useState<string | null>(null);
	const [entryContent, setEntryContent] = useState<string>("");
	const [viewMode, setViewMode] = useState<ViewMode>("notebook");
	const [selectedTag, setSelectedTag] = useState<string | null>(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	// Load content index
	useEffect(() => {
		fetch(`${import.meta.env.BASE_URL}research/content_index.json`)
			.then((res) => res.json())
			.then((data) => {
				setIndex(data);
				setLoading(false);
			})
			.catch((err) => {
				setError(`Failed to load research index: ${err.message}`);
				setLoading(false);
			});
	}, []);

	// Load selected entry content
	useEffect(() => {
		if (!selectedEntry) {
			setEntryContent("");
			return;
		}

		const path =
			viewMode === "notebook"
				? `${import.meta.env.BASE_URL}research/notebook/${selectedEntry}.md`
				: `${import.meta.env.BASE_URL}research/library/${selectedEntry}.md`;

		fetch(path)
			.then((res) => res.text())
			.then((content) => setEntryContent(content))
			.catch((err) => {
				setError(`Failed to load content: ${err.message}`);
				setEntryContent("# Error\n\nFailed to load content.");
			});
	}, [selectedEntry, viewMode]);

	if (loading) {
		return (
			<div className="min-h-screen bg-gray-900 text-white p-8">
				<div className="text-center">Loading research content...</div>
			</div>
		);
	}

	if (error || !index) {
		return (
			<div className="min-h-screen bg-gray-900 text-white p-8">
				<div className="text-center text-red-400">{error || "No index found"}</div>
			</div>
		);
	}

	const filteredEntries = selectedTag
		? index.notebook.entries.filter((e) => e.tags.includes(selectedTag))
		: index.notebook.entries;

	const filteredDocs = selectedTag
		? index.library.documents.filter((d) => d.category === selectedTag)
		: index.library.documents;

	return (
		<div className="min-h-screen bg-gray-900 text-white">
			{/* Header */}
			<div className="border-b border-gray-700 bg-gray-800">
				<div className="max-w-7xl mx-auto px-4 py-6">
					<h1 className="text-3xl font-bold mb-2">🔬 Research Browser</h1>
					<p className="text-gray-400">
						Explore our research journey - chronological notebook entries and comprehensive documentation
					</p>
				</div>
			</div>

			{/* View Mode Tabs */}
			<div className="border-b border-gray-700 bg-gray-800">
				<div className="max-w-7xl mx-auto px-4">
					<div className="flex gap-4">
						<button
							type="button"
							onClick={() => {
								setViewMode("notebook");
								setSelectedEntry(null);
								setSelectedTag(null);
							}}
							className={`px-6 py-3 font-semibold border-b-2 transition-colors ${
								viewMode === "notebook"
									? "border-blue-500 text-white"
									: "border-transparent text-gray-400 hover:text-white"
							}`}
						>
							📔 Notebook ({index.notebook.entries.length})
						</button>
						<button
							type="button"
							onClick={() => {
								setViewMode("library");
								setSelectedEntry(null);
								setSelectedTag(null);
							}}
							className={`px-6 py-3 font-semibold border-b-2 transition-colors ${
								viewMode === "library"
									? "border-blue-500 text-white"
									: "border-transparent text-gray-400 hover:text-white"
							}`}
						>
							📚 Library ({index.library.documents.length})
						</button>
					</div>
				</div>
			</div>

			<div className="max-w-7xl mx-auto px-4 py-8">
				<div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
					{/* Sidebar */}
					<div className="lg:col-span-1">
						{/* Description */}
						<div className="bg-gray-800 rounded-lg p-4 mb-4">
							<h3 className="font-semibold mb-2">
								{viewMode === "notebook" ? index.notebook.title : index.library.title}
							</h3>
							<p className="text-sm text-gray-400">
								{viewMode === "notebook" ? index.notebook.description : index.library.description}
							</p>
						</div>

						{/* Tag Filter */}
						{viewMode === "notebook" && (
							<div className="bg-gray-800 rounded-lg p-4 mb-4">
								<h4 className="font-semibold mb-2 text-sm">Filter by Tag</h4>
								<div className="flex flex-wrap gap-2">
									<button
										type="button"
										onClick={() => setSelectedTag(null)}
										className={`px-3 py-1 rounded-full text-xs ${
											!selectedTag
												? "bg-blue-600 text-white"
												: "bg-gray-700 text-gray-300 hover:bg-gray-600"
										}`}
									>
										All
									</button>
									{index.tags.map((tag) => (
										<button
											key={tag}
											type="button"
											onClick={() => setSelectedTag(tag)}
											className={`px-3 py-1 rounded-full text-xs ${
												selectedTag === tag
													? "bg-blue-600 text-white"
													: "bg-gray-700 text-gray-300 hover:bg-gray-600"
											}`}
										>
											{tag}
										</button>
									))}
								</div>
							</div>
						)}

						{/* Entry List */}
						<div className="bg-gray-800 rounded-lg">
							<div className="p-4 border-b border-gray-700">
								<h4 className="font-semibold text-sm">
									{viewMode === "notebook" ? "Recent Entries" : "Documents"}
								</h4>
							</div>
							<div className="divide-y divide-gray-700 max-h-[600px] overflow-y-auto">
								{viewMode === "notebook"
									? filteredEntries.map((entry) => (
											<button
												key={entry.id}
												type="button"
												onClick={() => setSelectedEntry(entry.id)}
												className={`w-full text-left p-4 hover:bg-gray-700 transition-colors ${
													selectedEntry === entry.id ? "bg-gray-700" : ""
												}`}
											>
												<div className="font-medium text-sm mb-1">{entry.title}</div>
												<div className="text-xs text-gray-400 mb-2">{entry.date}</div>
												{entry.status && (
													<div className="text-xs text-blue-400 mb-2">{entry.status}</div>
												)}
												<div className="flex flex-wrap gap-1 mb-2">
													{entry.tags.slice(0, 3).map((tag) => (
														<span
															key={tag}
															className="px-2 py-0.5 bg-gray-600 rounded text-xs"
														>
															{tag}
														</span>
													))}
												</div>
												<div className="text-xs text-gray-500">{entry.word_count} words</div>
											</button>
										))
									: filteredDocs.map((doc) => (
											<button
												key={doc.id}
												type="button"
												onClick={() => setSelectedEntry(doc.id)}
												className={`w-full text-left p-4 hover:bg-gray-700 transition-colors ${
													selectedEntry === doc.id ? "bg-gray-700" : ""
												}`}
											>
												<div className="font-medium text-sm mb-1">{doc.title}</div>
												<div className="text-xs text-gray-400 mb-1">
													{doc.category} • Modified {doc.modified}
												</div>
												<div className="text-xs text-gray-500">{doc.word_count} words</div>
											</button>
										))}
							</div>
						</div>
					</div>

					{/* Content Area */}
					<div className="lg:col-span-2">
						{selectedEntry && entryContent ? (
							<div className="bg-gray-800 rounded-lg p-8">
								<div className="prose prose-invert prose-blue max-w-none">
									<ReactMarkdown remarkPlugins={[remarkGfm]}>{entryContent}</ReactMarkdown>
								</div>
							</div>
						) : (
							<div className="bg-gray-800 rounded-lg p-8 text-center text-gray-400">
								<p className="text-lg mb-2">
									{viewMode === "notebook" ? "📔" : "📚"}
								</p>
								<p>
									Select {viewMode === "notebook" ? "a notebook entry" : "a document"} to read
								</p>
							</div>
						)}
					</div>
				</div>
			</div>
		</div>
	);
}
