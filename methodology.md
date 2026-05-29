# Document Layout Analyzer Methodology

This document outlines the architectural decisions, algorithms, and methodologies used in the `doc_layout_analyzer.py` script. The script is designed to process massive batches (e.g., 50,000+) of PDF documents, grouping them into unique layout templates while handling noise, variations in form fill-ins, and memory limitations natively.

## 1. Feature Extraction: Spatial & Semantic Fingerprints

To determine if two documents belong to the same template, the script analyzes two distinct dimensions:

*   **Spatial Dimension (Layout):** The script uses OCR to find the bounding boxes of every word on a page. These boxes are mapped onto a 2D grid. To make the matching robust against minor scanner shifts, a **Gaussian Blur** is applied to the grid, spreading the "weight" of each word into surrounding pixels. This creates a spatial heat map fingerprint.
*   **Semantic Dimension (Text):** The script extracts the raw text. Using a `TfidfVectorizer` (TF-IDF), it analyzes 1-gram to 3-gram phrases. By using specific frequency thresholds (`min_df`), it automatically ignores unique fill-ins (like a specific person's name) and focuses entirely on the boilerplate phrases (e.g., "Policy Number", "Date of Birth") that define the template.

Users can control the balance between these two dimensions using the `--semantic_weight` flag.

## 2. Memory Optimization: The Two-Stage Clustering Architecture

Computing a dense similarity matrix for 50,000 documents requires comparing every document to every other document (2.5 billion comparisons), which consumes ~80GB of RAM and causes Out of Memory (OOM) crashes on standard machines.

To solve this, the script uses a **Two-Stage Clustering Architecture**:

### Stage 1: The Greedy Leader Algorithm (Fast Deduplication)
Instead of a massive matrix, the script processes documents sequentially.
1. The first document becomes "Representative #1".
2. The next document is compared *only* to known Representatives.
3. If it is **>= 95% similar** to a Representative, it is grouped as a "child" of that template. The 95% threshold is intentional: it mathematically absorbs the 1% - 5% structural shifting caused by different people filling in the same form (e.g., a short name vs. a long name).
4. If it is less than 95% similar, it becomes a brand new Representative.

This reduces 50,000 documents down to perhaps 500 unique Representatives in seconds, dropping memory usage by 99% (Time Complexity: `O(N * K)`).

### Stage 2: Deep Analysis (Heavy Clustering)
The heavy algorithms are now run exclusively on the small subset of unique Representatives found in Stage 1.

Users can choose between two engines:
*   **Agglomerative Clustering:** Uses a strict distance threshold to merge templates. Good for controlled, well-understood document batches.
*   **HDBSCAN (Hierarchical Density-Based Spatial Clustering):** A dynamic engine that finds dense clusters automatically without needing a strict threshold. Crucially, it identifies unique outlier forms and isolates them into a "Noise" bucket, preventing them from contaminating clean template groups.

### Stage 3: Child Re-Mapping
Once the heavy clustering assigns final Group IDs to the Representatives, the script iterates through the Stage 1 map and assigns all the thousands of "child" documents to the final Group ID of their respective parent Representative.

## 3. Visualization & Output

To validate the clustering on massive datasets, the script outputs visual artifacts:
*   **Representative Heatmaps:** 2D visual representations of the blurred spatial grids for each unique template.
*   **Interactive HTML Dashboard:** Using `umap-learn` and Plotly, the high-dimensional fingerprints are reduced to 2D space. The script generates a local HTML web page allowing the user to visually explore the clusters, hover over dots to see text snippets, and read dynamic text-based recommendations for tuning parameters (e.g., adjusting `--blur` or `--min_cluster_size` based on the calculated noise ratio).