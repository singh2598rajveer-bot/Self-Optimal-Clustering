# Self-Optimal Image Clustering (SOC)

An advanced, mathematically optimized image segmentation algorithm implemented in Python. This project bridges the gap between theoretical research and functional software by translating the IEEE paper *"Self-Optimal Clustering Technique Using Optimized Threshold Function"* into a deployable machine learning pipeline.

## 🚀 Technical Highlights
* **Algorithmic Optimization:** Replaced heuristic guessing in standard mountain clustering with **Lagrange interpolation** to mathematically determine the optimal threshold function for cluster separation.
* **Memory-Efficient Architecture:** Engineered batch processing (chunking) to compute highly intensive $O(n^2)$ validation metrics—such as the Global Silhouette Index (GSI)—on massive image arrays without overloading system RAM.
* **Dynamic Sizing:** Capable of benchmarking RGB pixel data to dynamically identify the optimal number of clusters autonomously.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** NumPy, SciPy, Matplotlib, Pillow (PIL), tqdm

## 📊 Performance Metrics
The pipeline evaluates clustering quality using three distinct metrics, achieving high convergence and separation on multi-color test arrays:
1. **Global Silhouette Index (GSI)** *(Peak achieved: 0.868)*
2. **Partition Index (PI)**
3. **Separation Index (SI)**

## ⚙️ How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/Self-Optimal-Clustering.git](https://github.com/yourusername/Self-Optimal-Clustering.git)
   cd Self-Optimal-Clustering