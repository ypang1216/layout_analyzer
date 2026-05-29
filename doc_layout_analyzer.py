#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Document Layout Fingerprinting and Grouping Tool

This script analyzes a collection of PDF documents to identify and group them based
on the spatial layout of their text. It performs the following steps:

1.  **Pre-processing:** Converts initial pages of PDF files into images in parallel.
2.  **OCR (Optical Character Recognition):** Uses the `doctr` library to detect
    words and their bounding boxes in the images. This step is GPU-accelerated
    if a CUDA-enabled device is available.
3.  **Fingerprinting:** Creates a "spatial signature" or "fingerprint" for each
    document by mapping word centroids onto a 2D grid.
4.  **Grouping:** Compares the fingerprints using cosine similarity to group
    documents with similar layouts.
5.  **Reporting & Visualization:** Generates a summary report, detailed heatmaps for
    each document group, and a consolidated grid comparing the unique templates.

The script is designed as a command-line tool, configurable via arguments.

Usage:
    python your_script_name.py /path/to/pdfs /path/to/output --threshold 0.9 --pages 1

Dependencies can be installed via pip:
    pip install "doctr[torch]" PyMuPDF scikit-learn seaborn matplotlib tqdm
"""

import sys
import os
import glob
import logging
import time
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# --- Dependency Imports with Diagnostic Check ---
try:
    import torch
    import numpy as np
    import fitz  # PyMuPDF
    from PIL import Image
    from tqdm import tqdm
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import AgglomerativeClustering, HDBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer
    import scipy.ndimage
    import concurrent.futures
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    import umap
    from doctr.models import ocr_predictor
    from doctr.io import Document as DoctrDocument

except ImportError as e:
    print(f"\n--- Dependency Error ---")
    print(f"Failed to import a required library: {e.name}")
    print("Please ensure all dependencies are installed. You can install them using:")
    print("pip install \"doctr[torch]\" PyMuPDF scikit-learn scipy seaborn matplotlib tqdm plotly umap-learn")
    sys.exit(1)


# --- 1. Script Configuration ---
@dataclass
class Config:
    """Holds all configuration parameters for the script."""
    pdf_folder_path: str
    output_dir: str
    similarity_threshold: float = 0.9
    pages_to_process: int = 1
    grid_size: Tuple[int, int] = (10, 10)
    blur_sigma: float = 1.0  # Controls the spread of the Gaussian blur
    clustering_engine: str = "agglomerative"
    min_cluster_size: int = 5 # Used for HDBSCAN
    semantic_weight: float = 0.0 # Weight for text-based similarity
    max_workers: Optional[int] = None
    batch_size: int = 32
    comparison_grid_limit: int = 20
    log_level: str = "INFO"
    dpi: int = 96


# --- 2. Utility Functions ---
def setup_logging(level: str) -> None:
    """Configures the root logger."""
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s - %(levelname)s - %(message)s")
    # Suppress verbose logs from third-party libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

def perform_diagnostic_check() -> None:
    """Prints diagnostic information about the environment."""
    import doctr
    print("\n" + "--- DIAGNOSTIC INFORMATION ---".center(40))
    print(f"Python Executable: {sys.executable}")
    print(f"PyTorch version:     {torch.__version__}")
    print(f"doctr version:       {doctr.__version__}")
    print(f"CUDA Available:      {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:                 {torch.cuda.get_device_name(0)}")
    print("---".center(40) + "\n")

def process_pdf_to_images(pdf_path: str, pages_to_process: int, dpi: int) -> Optional[List[np.ndarray]]:
    """
    Converts leading pages of a PDF to a list of numpy arrays (images).

    Args:
        pdf_path: Path to the input PDF file.
        pages_to_process: The number of pages to convert from the start of the PDF.
        dpi: The resolution (dots per inch) to render the PDF page.

    Returns:
        A list of images as numpy arrays, or None if an error occurs.
    """
    try:
        doc = fitz.open(pdf_path)
        page_images = []
        num_pages_to_render = min(pages_to_process, doc.page_count)

        if num_pages_to_render == 0:
            doc.close()
            return []

        for i in range(num_pages_to_render):
            page = doc.load_page(i)
            # Render page to a pixmap at a configurable DPI for OCR
            pix = page.get_pixmap(dpi=dpi)
            # Convert to a PIL Image and then to a numpy array
            page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_images.append(np.array(page_image))
        doc.close()
        return page_images
    except Exception as e:
        logging.error(f"Failed to process {os.path.basename(pdf_path)}. Reason: {e}")
        return None


# --- 3. Core Fingerprinting and Visualization Logic ---
class DocumentFingerprinter:
    """
    Encapsulates the logic for document layout analysis, fingerprinting,
    and grouping.
    """
    def __init__(self, config: Config):
        """
        Initializes the fingerprinter with the given configuration.

        Args:
            config: A Config object containing all script settings.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self.pdf_files = self._get_pdf_files()
        self.model = ocr_predictor(pretrained=True).to(self.device)
        self.timings: Dict[str, float] = {}

    def _get_pdf_files(self) -> List[str]:
        """Validates and retrieves the list of PDF files to process."""
        pdf_files = glob.glob(os.path.join(self.config.pdf_folder_path, "*.pdf"))
        if not pdf_files:
            logging.error(f"No PDF files found in '{self.config.pdf_folder_path}'. Exiting.")
            sys.exit(1)
        logging.info(f"Found {len(pdf_files)} PDF files to analyze.")
        return pdf_files

    @staticmethod
    def _extract_page_data(page_result: Any) -> Tuple[List[List[float]], str]:
        """Extracts relative bounding box coordinates and raw text from a doctr Page result."""
        relative_boxes = []
        text_content = []
        for block in page_result.blocks:
            for line in block.lines:
                for word in line.words:
                    # word.geometry is ((x1, y1), (x2, y2))
                    relative_boxes.append([*word.geometry[0], *word.geometry[1]])
                    text_content.append(word.value)
        return relative_boxes, " ".join(text_content)

    @staticmethod
    def _create_spatial_signature(relative_boxes: List[List[float]], grid_size: Tuple[int, int], blur_sigma: float) -> np.ndarray:
        """Creates a flattened, blurred 2D histogram of word centroids."""
        rows, cols = grid_size
        signature = np.zeros((rows, cols), dtype=np.float32)
        if not relative_boxes:
            return signature.flatten()

        for x1, y1, x2, y2 in relative_boxes:
            centroid_x = (x1 + x2) / 2.0
            centroid_y = (y1 + y2) / 2.0
            col_idx = min(int(centroid_x * cols), cols - 1)
            row_idx = min(int(centroid_y * rows), rows - 1)
            signature[row_idx, col_idx] += 1

        # Apply Gaussian blur to spread the mass and make matching more robust
        blurred_signature = scipy.ndimage.gaussian_filter(signature, sigma=blur_sigma)
        return blurred_signature.flatten()

    def _preprocess_pdfs(self) -> Tuple[List[List[np.ndarray]], List[int]]:
        """
        Stage 1: Parallel Pre-processing (CPU-Bound). Converts PDFs to images.
        """
        logging.info(f"Starting PDF pre-processing on up to {os.cpu_count()} cores.")
        all_docs_pages = []
        doc_page_counts = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Create a map of futures to keep track of PDF paths
            future_to_pdf = {
                executor.submit(process_pdf_to_images, pdf, self.config.pages_to_process, self.config.dpi): pdf
                for pdf in self.pdf_files
            }
            pbar = tqdm(concurrent.futures.as_completed(future_to_pdf), total=len(self.pdf_files), desc="Pre-processing PDFs")
            for future in pbar:
                res = future.result()
                if res is not None:
                    all_docs_pages.append(res)
                    doc_page_counts.append(len(res))
                else:
                    # Handle processing failure by appending empty lists to maintain order
                    all_docs_pages.append([])
                    doc_page_counts.append(0)

        return all_docs_pages, doc_page_counts

    def _run_ocr(self, all_docs_pages: List[List[np.ndarray]]) -> DoctrDocument:
        """Stage 2: Batched OCR (GPU-Bound)."""
        logging.info("Pre-processing complete. Starting batched OCR...")
        flat_page_list = [page for doc_pages in all_docs_pages for page in doc_pages]

        if not flat_page_list:
            logging.error("No pages could be extracted from any PDF. Exiting.")
            sys.exit(1)

        all_pages_results = []
        pbar = tqdm(total=len(flat_page_list), desc="Running OCR in batches")
        for i in range(0, len(flat_page_list), self.config.batch_size):
            batch = flat_page_list[i : i + self.config.batch_size]
            # The model returns a Document object which contains the pages
            batch_result_doc = self.model(batch)
            # Access the .pages attribute which is the iterable list
            all_pages_results.extend(batch_result_doc.pages)
            pbar.update(len(batch))
        pbar.close()

        # Reconstruct the Doctr Document object correctly
        return DoctrDocument(pages=all_pages_results)


    def _generate_fingerprints(self, ocr_results: DoctrDocument, doc_page_counts: List[int]) -> Tuple[List[Optional[np.ndarray]], List[str]]:
        """Stage 3: Fingerprint Generation & Text Extraction (Fast CPU work)."""
        logging.info("OCR complete. Generating spatial fingerprints and extracting text...")
        all_doc_fingerprints = []
        all_doc_texts = []
        current_pos = 0

        for page_count in tqdm(doc_page_counts, desc="Generating Fingerprints"):
            if page_count == 0:
                all_doc_fingerprints.append(None)
                all_doc_texts.append("")
                continue

            doc_ocr_pages = ocr_results.pages[current_pos : current_pos + page_count]
            current_pos += page_count
            page_fingerprints = []
            page_texts = []

            for page_result in doc_ocr_pages:
                relative_boxes, text = self._extract_page_data(page_result)
                page_fingerprints.append(self._create_spatial_signature(relative_boxes, self.config.grid_size, self.config.blur_sigma))
                page_texts.append(text)

            # Sum fingerprints of all pages in a document to create a single doc fingerprint
            doc_fingerprint = np.sum(page_fingerprints, axis=0)
            all_doc_fingerprints.append(doc_fingerprint)
            all_doc_texts.append(" ".join(page_texts))

        return all_doc_fingerprints, all_doc_texts

    def _group_documents(self, all_doc_fingerprints: List[Optional[np.ndarray]], all_doc_texts: List[str]) -> Tuple[List[np.ndarray], List[List[str]]]:
        """Stage 4: Similarity Comparison and Grouping (Clustering)."""
        logging.info("Fingerprinting complete. Grouping documents by similarity...")

        # Filter out empty/invalid fingerprints and keep track of valid indices
        valid_indices = []
        valid_fps = []
        valid_texts = []
        for i, fp in enumerate(all_doc_fingerprints):
            if fp is not None and np.sum(fp) > 0:
                valid_indices.append(i)
                valid_fps.append(fp)
                valid_texts.append(all_doc_texts[i])
            else:
                pdf_name = os.path.basename(self.pdf_files[i])
                logging.warning(f"Skipping '{pdf_name}' due to empty or invalid fingerprint.")

        if not valid_fps:
            return [], []

        import gc
        # Convert to float32 to save memory (cuts memory usage in half for distance matrices)
        valid_fps_matrix = np.array(valid_fps, dtype=np.float32)

        # If there's only 1 valid document, just return it as a single group
        if len(valid_fps) == 1:
            return [valid_fps[0]], [[os.path.basename(self.pdf_files[valid_indices[0]])]]

        num_docs = len(valid_fps)

        # --- Stage 1: Fast Pre-Clustering (Greedy Leader Algorithm) ---
        # Instead of computing an N x N matrix, we maintain a small list of unique "Template Representatives".
        # As we process each document, we compare it ONLY to the known representatives.
        # If it is >= 95% similar to a representative, we assign it as a "child" of that template.
        # This >95% threshold perfectly absorbs the 1-5% noise introduced by different people filling in forms
        # (names, dates, check boxes) while keeping computational and memory complexity extremely low (O(N * K)).
        logging.info(f"Stage 1: Pre-clustering {num_docs} documents to find unique representatives (absorbs fill-ins)...")

        rep_to_children = {}  # Map: representative index -> list of child indices
        representative_indices = []
        representative_fps = []

        # We use a very strict threshold for Stage 1.
        # Anything lower will be evaluated in Stage 2's heavy clustering.
        greedy_threshold = 0.95

        for idx in range(num_docs):
            current_fp = valid_fps_matrix[idx]

            if not representative_indices:
                # First document becomes the first representative
                representative_indices.append(idx)
                representative_fps.append(current_fp)
                rep_to_children[idx] = [idx]
                continue

            # Compare current document against ALL known representatives simultaneously
            # cosine_similarity expects 2D arrays, so we reshape current_fp
            sim_scores = cosine_similarity(current_fp.reshape(1, -1), np.array(representative_fps))[0]

            best_match_idx = np.argmax(sim_scores)
            best_score = sim_scores[best_match_idx]

            if best_score >= greedy_threshold:
                # It's an exact or highly similar template (just filled in differently)
                rep_idx = representative_indices[best_match_idx]
                rep_to_children[rep_idx].append(idx)
            else:
                # It's a completely new template
                representative_indices.append(idx)
                representative_fps.append(current_fp)
                rep_to_children[idx] = [idx]

        num_reps = len(representative_indices)
        logging.info(f"Pre-clustering complete: Reduced {num_docs} documents to {num_reps} unique representatives.")

        if num_reps > 10000:
            logging.warning(f"Processing {num_reps} unique representatives. Computing N x N dense matrices may consume significant memory.")

        # Extract only the representatives for the heavy N x N math
        rep_fps_matrix = valid_fps_matrix[representative_indices]
        rep_texts = [valid_texts[i] for i in representative_indices]

        # --- Stage 2: Heavy Semantic/Spatial Clustering ---
        # 1. Compute SPATIAL distance matrix for representatives (in-place memory optimization)
        distance_matrix = cosine_similarity(rep_fps_matrix)
        # Convert similarity (1.0) to distance (0.0) in place
        np.subtract(1.0, distance_matrix, out=distance_matrix)
        np.clip(distance_matrix, 0.0, 2.0, out=distance_matrix)

        # 2. Compute SEMANTIC distance matrix for representatives (if requested)
        if self.config.semantic_weight > 0.0:
            logging.info(f"Computing semantic distances (weight: {self.config.semantic_weight})")
            try:
                # min_df ignores unique fill-ins, ngram_range captures boilerplate phrases
                vectorizer = TfidfVectorizer(min_df=0.05, max_df=1.0, ngram_range=(1, 3))
                tfidf_matrix = vectorizer.fit_transform(rep_texts).astype(np.float32)

                # Compute semantic distance in a new array, but do it in-place
                semantic_distance = cosine_similarity(tfidf_matrix)
                np.subtract(1.0, semantic_distance, out=semantic_distance)
                np.clip(semantic_distance, 0.0, 2.0, out=semantic_distance)

                # Combine matrices via weighted average in-place onto distance_matrix
                spatial_weight = 1.0 - self.config.semantic_weight
                np.multiply(distance_matrix, spatial_weight, out=distance_matrix)
                np.multiply(semantic_distance, self.config.semantic_weight, out=semantic_distance)
                np.add(distance_matrix, semantic_distance, out=distance_matrix)

                # Free memory explicitly
                del semantic_distance
                del tfidf_matrix
                gc.collect()
            except ValueError as e:
                # E.g., if vocabulary is empty because all words were filtered out
                logging.warning(f"Semantic processing failed, falling back to spatial only. Reason: {e}")

        if self.config.clustering_engine == "agglomerative":
            logging.info("Using Agglomerative Clustering engine.")
            distance_threshold = 1.0 - self.config.similarity_threshold
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="average",
                distance_threshold=distance_threshold
            )
            labels = clustering.fit_predict(distance_matrix)
        elif self.config.clustering_engine == "hdbscan":
            logging.info("Using HDBSCAN Clustering engine.")
            clustering = HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                metric="precomputed",
                cluster_selection_epsilon=1.0 - self.config.similarity_threshold # Optional: allow merging closer clusters
            )
            labels = clustering.fit_predict(distance_matrix)
        else:
            raise ValueError(f"Unknown clustering engine: {self.config.clustering_engine}")

        # Determine the number of valid clusters (excluding noise, which is -1 in HDBSCAN)
        unique_labels = set(labels)
        has_noise = -1 in unique_labels
        num_clusters = len(unique_labels) - (1 if has_noise else 0)

        unique_template_fingerprints = []
        # Initialize groups. If there's noise, we'll append a special 'Noise' group at the end
        grouped_files = [[] for _ in range(num_clusters)]
        noise_files = []

        # --- Stage 3: Map all children back to their final cluster IDs ---
        # `labels` only contains the cluster IDs for the `num_reps` representatives.
        # We need to map every original document back to the cluster of its representative.
        for rep_enum_idx, label in enumerate(labels):
            rep_original_idx = representative_indices[rep_enum_idx]

            # Get all child document indices that belong to this representative
            child_indices = rep_to_children[rep_original_idx]

            for child_idx in child_indices:
                global_pdf_idx = valid_indices[child_idx]
                pdf_name = os.path.basename(self.pdf_files[global_pdf_idx])

                if label == -1:
                    noise_files.append(pdf_name)
                else:
                    grouped_files[label].append(pdf_name)

        # Calculate representative fingerprints for each valid cluster (e.g. mean of the representatives)
        for label in range(num_clusters):
            # Get indices of the representatives that belong to this cluster
            cluster_rep_indices = np.where(labels == label)[0]
            cluster_fps = rep_fps_matrix[cluster_rep_indices]
            representative_fp = np.mean(cluster_fps, axis=0)
            unique_template_fingerprints.append(representative_fp)

        if noise_files:
            logging.info(f"HDBSCAN marked {len(noise_files)} documents as noise/outliers.")
            # We don't generate a single representative fingerprint for noise since they are diverse
            # but we can append them to the grouped files for reporting
            grouped_files.append(noise_files)
            # Add a zero fingerprint for noise so lists remain aligned, though we might skip it in visualization
            unique_template_fingerprints.append(np.zeros_like(valid_fps_matrix[0]))

        return unique_template_fingerprints, grouped_files


    def _generate_visual_reports(self, grouped_files: List[List[str]], unique_template_fps: List[np.ndarray], all_doc_fingerprints: List[Optional[np.ndarray]]) -> None:
        """Stage 5: Generate Comparison Grid and Individual Heatmaps."""
        if not grouped_files:
            logging.warning("No groups were formed. Skipping report generation.")
            return

        logging.info("Generating visual validation reports...")

        # Create a mapping from a filename to its fingerprint for easy lookup
        fingerprint_map = {os.path.basename(self.pdf_files[i]): fp for i, fp in enumerate(all_doc_fingerprints)}

        # Sort groups by size (most common templates first) for reporting
        sorted_groups = sorted(grouped_files, key=len, reverse=True)

        # 1. Generate the main comparison grid
        self._generate_comparison_grid_image(sorted_groups, fingerprint_map)

        # 2. Generate individual heatmaps for each group
        pbar_desc = "Generating Individual Heatmaps"
        for i, files in enumerate(tqdm(sorted_groups, desc=pbar_desc)):
            template_index = i + 1
            template_dir = os.path.join(self.config.output_dir, f"Template_{template_index:03d}")
            os.makedirs(template_dir, exist_ok=True)

            # Generate heatmap for the representative (first) file of the template
            main_file_name = files[0]
            self._save_heatmap(
                fingerprint=fingerprint_map.get(main_file_name),
                title=f"Template #{template_index}: {main_file_name}",
                output_path=os.path.join(template_dir, f"TEMPLATE_{main_file_name}.png")
            )

            # Generate heatmaps for a few other files in the same group for comparison
            for similar_file_name in files[1:4]: # Save up to 3 similar examples
                self._save_heatmap(
                    fingerprint=fingerprint_map.get(similar_file_name),
                    title=f"Similar to T#{template_index}: {similar_file_name}",
                    output_path=os.path.join(template_dir, f"SIMILAR_{similar_file_name}.png")
                )


    def _generate_comparison_grid_image(self, sorted_groups: List[List[str]], fingerprint_map: Dict[str, np.ndarray]) -> None:
        """Helper to create and save the multi-template comparison grid image."""
        templates_to_plot = sorted_groups[:self.config.comparison_grid_limit]
        if not templates_to_plot:
            return

        num_templates = len(templates_to_plot)
        ncols = 5
        nrows = math.ceil(num_templates / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), squeeze=False)
        fig.suptitle(f"Top {num_templates} Unique Document Templates", fontsize=20, y=0.98)

        for i, group in enumerate(templates_to_plot):
            row, col = divmod(i, ncols)
            ax = axes[row, col]
            
            # Get fingerprint of the representative file
            representative_fp = fingerprint_map.get(group[0])
            if representative_fp is None: continue

            heatmap_data = representative_fp.reshape(self.config.grid_size)
            sns.heatmap(heatmap_data, annot=False, cmap="viridis", cbar=False, ax=ax, square=True)
            title = f"Template #{i+1} ({len(group)} files)\n{os.path.basename(group[0])}"
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        for i in range(num_templates, nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row, col].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = os.path.join(self.config.output_dir, "template_comparison_grid.png")
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logging.info(f"Saved template comparison grid to: {output_path}")

    def _generate_interactive_dashboard(self, all_doc_fingerprints: List[Optional[np.ndarray]], all_doc_texts: List[str], grouped_files: List[List[str]], unique_template_fps: List[np.ndarray]) -> None:
        """Generates an interactive Plotly dashboard for exploring the clusters."""
        logging.info("Generating interactive HTML dashboard...")

        valid_indices = []
        valid_fps = []
        hover_texts = []
        cluster_labels = []

        # Build lookup maps
        file_to_cluster = {}
        for cluster_id, files in enumerate(grouped_files):
            # If HDBSCAN is used, the last group is noise ONLY IF it was marked as such (zeros fingerprint)
            is_noise = False
            if self.config.clustering_engine == "hdbscan" and cluster_id == len(grouped_files) - 1:
                if len(unique_template_fps) > cluster_id and np.sum(unique_template_fps[cluster_id]) == 0:
                    is_noise = True

            label_name = "Noise / Outliers" if is_noise else f"Template #{cluster_id + 1}"
            for f in files:
                file_to_cluster[f] = label_name

        for i, fp in enumerate(all_doc_fingerprints):
            if fp is not None and np.sum(fp) > 0:
                valid_indices.append(i)
                valid_fps.append(fp)
                filename = os.path.basename(self.pdf_files[i])
                text_snippet = all_doc_texts[i][:200] + "..." if len(all_doc_texts[i]) > 200 else all_doc_texts[i]
                hover_texts.append(f"<b>File:</b> {filename}<br><b>Text Snippet:</b><br>{text_snippet}")
                cluster_labels.append(file_to_cluster.get(filename, "Unknown"))

        if len(valid_fps) < 3:
            logging.warning("Not enough valid documents to generate UMAP visualization.")
            return

        # Use UMAP to reduce the high-dimensional fingerprints to 2D for scatter plotting
        # Note: setting n_neighbors safely based on dataset size
        n_neighbors = min(15, max(2, len(valid_fps) - 1))
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(np.array(valid_fps))

        # Create Plotly figure
        fig = px.scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            color=cluster_labels,
            hover_name=hover_texts,
            title="Document Cluster Visualization (UMAP)",
            labels={'color': 'Cluster Group'}
        )

        # Generate Dynamic Insights
        num_docs = len(valid_fps)
        noise_docs = cluster_labels.count("Noise / Outliers")
        noise_ratio = (noise_docs / num_docs) * 100 if num_docs > 0 else 0

        insights_html = f"<h3>Analysis Insights</h3><ul>"
        insights_html += f"<li><b>Total Documents Analyzed:</b> {num_docs}</li>"

        if self.config.clustering_engine == "hdbscan":
            insights_html += f"<li><b>Noise Ratio:</b> {noise_ratio:.1f}% ({noise_docs} outliers)</li>"
            if noise_ratio > 15:
                insights_html += "<li><span style='color:red;'><b>Recommendation:</b> Noise ratio is high (>15%). Consider increasing <code>--blur</code> to be more forgiving of layout shifts, or decrease <code>--min_cluster_size</code> to allow smaller identical groups to form instead of being marked as noise.</span></li>"
            elif noise_ratio < 2:
                insights_html += "<li><span style='color:green;'><b>Recommendation:</b> Noise ratio is very low. Clusters look well-defined. If templates are incorrectly merged, consider lowering <code>--blur</code> or increasing the <code>--semantic_weight</code>.</span></li>"
        else:
            insights_html += f"<li><b>Engine:</b> Agglomerative Clustering (Strict thresholding).</li>"
            insights_html += "<li><span style='color:blue;'><b>Recommendation:</b> Agglomerative clustering does not flag noise. If you are seeing vastly different templates merged together, increase your <code>--threshold</code> closer to 1.0. If you are running massive batches, consider switching to <code>--engine hdbscan</code> to automatically isolate unique outlier forms.</span></li>"

        insights_html += "</ul>"

        # Write to HTML
        output_path = os.path.join(self.config.output_dir, "interactive_dashboard.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("<html><head><title>Layout Analyzer Dashboard</title>")
            f.write("<style>body { font-family: sans-serif; margin: 20px; }</style></head><body>")
            f.write("<h1>Document Layout Grouping Dashboard</h1>")
            f.write(insights_html)
            f.write("<hr>")
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("</body></html>")

        logging.info(f"Saved interactive dashboard to: {output_path}")

    def _save_heatmap(self, fingerprint: Optional[np.ndarray], title: str, output_path: str) -> None:
        """Generates and saves a single heatmap from a flattened fingerprint."""
        if fingerprint is None or np.sum(fingerprint) == 0:
            logging.warning(f"Skipping heatmap for '{title}' due to empty fingerprint.")
            return

        try:
            heatmap_data = fingerprint.reshape(self.config.grid_size)
            plt.figure(figsize=(8, 8))
            sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="viridis", cbar=True, linewidths=.5, square=True)
            plt.title(title, fontsize=14)
            plt.xlabel("Grid Columns")
            plt.ylabel("Grid Rows")
            plt.tight_layout()
            plt.savefig(output_path, dpi=100)
            plt.close()
        except Exception as e:
            logging.error(f"Could not generate heatmap for '{title}'. Reason: {e}")

    def _print_final_report(self, grouped_files: List[List[str]]) -> None:
        """Prints the performance timings and a summary of the grouped files."""
        # Performance Report
        print("\n" + "="*28)
        print("   PERFORMANCE REPORT")
        print("="*28)
        for stage, duration in self.timings.items():
            print(f"{stage:<28}: {duration:.2f} seconds")
        print("="*28 + "\n")

        # Analysis Summary
        print("--- Analysis Complete ---")
        if not grouped_files:
            print("Could not generate a valid fingerprint for any document.")
        else:
            print(f"Found {len(grouped_files)} unique document templates.")
            print(f"✅ Visual validation reports saved to: '{self.config.output_dir}'\n")

            # Sort groups by size, but keep the 'Noise' group (if it exists) at the very end
            valid_groups = [g for g in grouped_files if self.config.clustering_engine != "hdbscan" or grouped_files.index(g) != len(grouped_files) - 1 or len(unique_template_fingerprints) == 0 or np.sum(unique_template_fingerprints[-1]) != 0]
            noise_group = grouped_files[-1] if self.config.clustering_engine == "hdbscan" and grouped_files and np.sum(unique_template_fingerprints[-1]) == 0 else []

            sorted_valid_groups = sorted(valid_groups, key=len, reverse=True)

            for i, files in enumerate(sorted_valid_groups):
                print(f"📄 Template #{i + 1} ({len(files)} files):")
                print(f"   - {files[0]} (Representative)")
                # Print up to 5 similar files for brevity
                for file_name in sorted(files[1:6]):
                    print(f"   - {file_name}")
                if len(files) > 6:
                    print(f"   ... and {len(files) - 6} more.")
                print("-" * 25)

            if noise_group:
                print(f"⚠️  Noise / Outliers ({len(noise_group)} files):")
                for file_name in sorted(noise_group[:5]):
                    print(f"   - {file_name}")
                if len(noise_group) > 5:
                    print(f"   ... and {len(noise_group) - 5} more.")
                print("-" * 25)

    def run(self) -> None:
        """Executes the full document fingerprinting and grouping pipeline."""
        script_start_time = time.monotonic()
        os.makedirs(self.config.output_dir, exist_ok=True)

        # --- Pipeline Stages ---
        stage_time = time.monotonic()
        all_docs_pages, doc_page_counts = self._preprocess_pdfs()
        self.timings["1. PDF Pre-processing (CPU)"] = time.monotonic() - stage_time

        stage_time = time.monotonic()
        ocr_results = self._run_ocr(all_docs_pages)
        self.timings["2. OCR Processing (GPU/CPU)"] = time.monotonic() - stage_time

        stage_time = time.monotonic()
        all_doc_fingerprints, all_doc_texts = self._generate_fingerprints(ocr_results, doc_page_counts)
        self.timings["3. Fingerprint & Text Ext."] = time.monotonic() - stage_time

        stage_time = time.monotonic()
        unique_fps, grouped_files = self._group_documents(all_doc_fingerprints, all_doc_texts)
        self.timings["4. Similarity Grouping"] = time.monotonic() - stage_time

        stage_time = time.monotonic()
        self._generate_visual_reports(grouped_files, unique_fps, all_doc_fingerprints)
        self._generate_interactive_dashboard(all_doc_fingerprints, all_doc_texts, grouped_files, unique_fps)
        self.timings["5. Visual Report Generation"] = time.monotonic() - stage_time
        
        self.timings["Total Script Runtime"] = time.monotonic() - script_start_time

        # --- Final Reporting ---
        self._print_final_report(grouped_files)


def main():
    """Main execution function."""
    # Check for the diagnostic flag before full argument parsing
    if '--diag' in sys.argv:
        perform_diagnostic_check()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Group PDF documents by visual layout similarity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("pdf_folder", help="Path to the folder containing PDF files.")
    parser.add_argument("output_folder", help="Path to the folder where results will be saved.")
    parser.add_argument("-t", "--threshold", type=float, default=0.9,
                        help="Cosine similarity threshold for grouping documents (0.0 to 1.0).")
    parser.add_argument("-p", "--pages", type=int, default=1,
                        help="Number of pages to process from the beginning of each PDF.")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Batch size for OCR processing on the GPU.")
    parser.add_argument("--dpi", type=int, default=96,
                        help="Dots Per Inch (DPI) to use when rendering PDF pages to images.")
    parser.add_argument("--blur", type=float, default=1.0,
                        help="Gaussian blur sigma for spatial fingerprints. Higher values allow more flexibility in layout matches.")
    parser.add_argument("--profile", choices=["loss_runs", "applications", "default"], default="default",
                        help="Select a pre-configured tuning profile for specific document types. Overrides --blur and --threshold.")
    parser.add_argument("--engine", choices=["agglomerative", "hdbscan"], default="agglomerative", dest="clustering_engine",
                        help="Choose the clustering algorithm. 'agglomerative' uses a strict threshold. 'hdbscan' auto-finds dense clusters and isolates noise.")
    parser.add_argument("--min_cluster_size", type=int, default=5,
                        help="Minimum number of documents to form a cluster (only used if --engine is hdbscan).")
    parser.add_argument("--semantic_weight", type=float, default=0.0,
                        help="Weight (0.0 to 1.0) given to text content similarity vs spatial layout. 0.0 = purely spatial, 0.5 = 50/50 mix, 1.0 = purely semantic.")
    # The --diag action is now handled manually above, but we keep it for --help message
    parser.add_argument("--diag", action="store_true", help="Run a diagnostic check and exit.")

    args = parser.parse_args()

    # --- Profile Tuning Logic ---
    threshold = args.threshold
    blur = args.blur

    if args.profile == "loss_runs":
        threshold = 0.95
        blur = 0.5
        print("Using 'loss_runs' profile: Setting threshold=0.95, blur=0.5")
    elif args.profile == "applications":
        threshold = 0.85
        blur = 1.5
        print("Using 'applications' profile: Setting threshold=0.85, blur=1.5")
    elif args.profile == "default":
        # Do not override, use explicitly passed args or defaults
        pass

    # --- Setup ---
    setup_logging("INFO")
    # No need to call perform_diagnostic_check() here again as it's handled above
    
    # --- Configuration ---
    config = Config(
        pdf_folder_path=args.pdf_folder,
        output_dir=args.output_folder,
        similarity_threshold=threshold,
        pages_to_process=args.pages,
        batch_size=args.batch_size,
        dpi=args.dpi,
        blur_sigma=blur,
        clustering_engine=args.clustering_engine,
        min_cluster_size=args.min_cluster_size,
        semantic_weight=args.semantic_weight,
    )

    # --- Execution ---
    fingerprinter = DocumentFingerprinter(config)
    fingerprinter.run()


if __name__ == "__main__":
    main()
