import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from SelfOptimizedClustering import MountainClusteringSOC
from MyMetrics import global_silhouette_score, partition_index, separation_index












BASE_DIR    = "."
RESULTS_DIR = "./results"
IMAGES      = ["image1.jpg", "image2.jpg", "image3.jpg"]
M_VALUES    = [2, 3, 4, 5]





def load_image(path: str):
    img = Image.open(path).convert("RGB")
    img.thumbnail((256, 256))                                                      ## resize for speed (GSI is O(n^2))
    arr = np.array(img, dtype=np.float64); H, W, _ = arr.shape                     ## H x W x 3
    return arr.reshape(-1, 3).T, (H, W)                                            ## 3 x (H*W), d=3  n=H*W

def save_image(centers: np.ndarray, y: np.ndarray, shape: tuple, path: str):
    H, W = shape
    segmented = np.clip(centers[y].reshape(H, W, 3), 0, 255).astype(np.uint8)      ## map each pixel to its cluster center
    Image.fromarray(segmented).save(path)

def plot_gsi(img_name: str, hist_gsi: list):
    best_idx = np.argmax(hist_gsi)
    plt.figure(figsize=(8, 6))
    plt.plot(M_VALUES, hist_gsi, color='red', linestyle='--', marker='s', mfc='green')
    plt.plot(M_VALUES[best_idx], hist_gsi[best_idx], marker='s', color='black')
    plt.annotate(f"X: {M_VALUES[best_idx]}\nY: {hist_gsi[best_idx]:.4f}",
                 xy=(M_VALUES[best_idx], hist_gsi[best_idx]), xytext=(10, -30),
                 textcoords='offset points', bbox=dict(boxstyle="square,pad=0.3", fc="lightyellow", ec="silver"))
    plt.xlabel('Number of clusters'); plt.ylabel('Global Silhouette Index (GSI)')
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, f"GSI_vs_M_{img_name}.png")); plt.close()

def evaluate():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "evaluation_metrics.txt"), "w") as f:
        for img_name in IMAGES:
            img_path = os.path.join(BASE_DIR, img_name)
            if not os.path.exists(img_path): continue
            X, shape = load_image(img_path)
            Xmin = X.min(axis=1, keepdims=True); Xmax = X.max(axis=1, keepdims=True)    
            denom = Xmax - Xmin; denom[denom == 0] = 1.0
            best_gsi, best_y, best_centers = -1.0, None, None
            hist_gsi = []
            f.write(f"Results for {img_name}:\n")
            f.write("M | GSI      | PI       | SI      \n")
            f.write("-" * 38 + "\n"); f.flush()
            for M in M_VALUES:
                soc = MountainClusteringSOC(X, no_of_clusters = M)                    
                soc.train(iterations = 10)                                              
                y = soc.predict(soc.X)
                gsi = global_silhouette_score(soc.X, y, M)                             
                pi  = partition_index(soc.X, y, M)
                si  = separation_index(soc.X, y, M)
                hist_gsi.append(gsi)
                f.write(f"{M} | {gsi:.4f} | {pi:.4f} | {si:.4f}\n"); f.flush()
                if gsi > best_gsi:
                    best_gsi = gsi; best_y = y
                    best_centers = soc.CENTERS * denom.T + Xmin.T                       
            f.write("\n"); f.flush()
            save_image(best_centers, best_y, shape, os.path.join(RESULTS_DIR, f"segmented_{img_name}"))
            plot_gsi(img_name, hist_gsi)

if __name__ == "__main__":
    evaluate()
