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
    import concurrent.futures
    import matplotlib.pyplot as plt
    import seaborn as sns
    from doctr.models import ocr_predictor
    from doctr.io import Document as DoctrDocument

except ImportError as e:
    print(f"\n--- Dependency Error ---")
    print(f"Failed to import a required library: {e.name}")
    print("Please ensure all dependencies are installed. You can install them using:")
    print("pip install \"doctr[torch]\" PyMuPDF scikit-learn seaborn matplotlib tqdm")
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
    def _extract_relative_boxes(page_result: Any) -> List[List[float]]:
        """Extracts relative bounding box coordinates from a doctr Page result."""
        relative_boxes = []
        for block in page_result.blocks:
            for line in block.lines:
                for word in line.words:
                    # word.geometry is ((x1, y1), (x2, y2))
                    relative_boxes.append([*word.geometry[0], *word.geometry[1]])
        return relative_boxes

    @staticmethod
    def _create_spatial_signature(relative_boxes: List[List[float]], grid_size: Tuple[int, int]) -> np.ndarray:
        """Creates a flattened 2D histogram of word centroids."""
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
        return signature.flatten()

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


    def _generate_fingerprints(self, ocr_results: DoctrDocument, doc_page_counts: List[int]) -> List[Optional[np.ndarray]]:
        """Stage 3: Fingerprint Generation (Fast CPU work)."""
        logging.info("OCR complete. Generating fingerprints...")
        all_doc_fingerprints = []
        current_pos = 0

        for page_count in tqdm(doc_page_counts, desc="Generating Fingerprints"):
            if page_count == 0:
                all_doc_fingerprints.append(None)
                continue

            doc_ocr_pages = ocr_results.pages[current_pos : current_pos + page_count]
            current_pos += page_count
            page_fingerprints = []
            for page_result in doc_ocr_pages:
                relative_boxes = self._extract_relative_boxes(page_result)
                page_fingerprints.append(self._create_spatial_signature(relative_boxes, self.config.grid_size))

            # Sum fingerprints of all pages in a document to create a single doc fingerprint
            doc_fingerprint = np.sum(page_fingerprints, axis=0)
            all_doc_fingerprints.append(doc_fingerprint)

        return all_doc_fingerprints

    def _group_documents(self, all_doc_fingerprints: List[Optional[np.ndarray]]) -> Tuple[List[np.ndarray], List[List[str]]]:
        """Stage 4: Similarity Comparison and Grouping."""
        logging.info("Fingerprinting complete. Grouping documents by similarity...")
        unique_template_fingerprints: List[np.ndarray] = []
        grouped_files: List[List[str]] = []

        for i, current_fp in enumerate(tqdm(all_doc_fingerprints, desc="Grouping Documents")):
            pdf_name = os.path.basename(self.pdf_files[i])
            if current_fp is None or np.sum(current_fp) == 0:
                logging.warning(f"Skipping '{pdf_name}' due to empty or invalid fingerprint.")
                continue

            found_match = False
            for j, existing_fp in enumerate(unique_template_fingerprints):
                # Reshape for scikit-learn's cosine_similarity function
                score = cosine_similarity(current_fp.reshape(1, -1), existing_fp.reshape(1, -1))[0][0]
                if score > self.config.similarity_threshold:
                    grouped_files[j].append(pdf_name)
                    found_match = True
                    break

            if not found_match:
                unique_template_fingerprints.append(current_fp)
                grouped_files.append([pdf_name])

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

            sorted_groups = sorted(grouped_files, key=len, reverse=True)
            for i, files in enumerate(sorted_groups):
                print(f"📄 Template #{i + 1} ({len(files)} files):")
                print(f"   - {files[0]} (Representative)")
                # Print up to 5 similar files for brevity
                for file_name in sorted(files[1:6]):
                    print(f"   - {file_name}")
                if len(files) > 6:
                    print(f"   ... and {len(files) - 6} more.")
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
        all_doc_fingerprints = self._generate_fingerprints(ocr_results, doc_page_counts)
        self.timings["3. Fingerprint Generation"] = time.monotonic() - stage_time

        stage_time = time.monotonic()
        unique_fps, grouped_files = self._group_documents(all_doc_fingerprints)
        self.timings["4. Similarity Grouping"] = time.monotonic() - stage_time

        stage_time = time.monotonic()
        self._generate_visual_reports(grouped_files, unique_fps, all_doc_fingerprints)
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
    # The --diag action is now handled manually above, but we keep it for --help message
    parser.add_argument("--diag", action="store_true", help="Run a diagnostic check and exit.")

    args = parser.parse_args()

    # --- Setup ---
    setup_logging("INFO")
    # No need to call perform_diagnostic_check() here again as it's handled above
    
    # --- Configuration ---
    config = Config(
        pdf_folder_path=args.pdf_folder,
        output_dir=args.output_folder,
        similarity_threshold=args.threshold,
        pages_to_process=args.pages,
        batch_size=args.batch_size,
        dpi=args.dpi,
    )

    # --- Execution ---
    fingerprinter = DocumentFingerprinter(config)
    fingerprinter.run()


if __name__ == "__main__":
    main()
