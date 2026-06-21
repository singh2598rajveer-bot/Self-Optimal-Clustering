import numpy as np
from scipy.interpolate import lagrange
from tqdm import tqdm






















class MountainClusteringSOC:

    def __init__(self, X: np.ndarray, no_of_clusters: int) -> None:
        self.X = X.copy(); self.normalize()                                                 ## datapoints, size d x n, dimension d, n points
        self.M = no_of_clusters                                                             ## IMMUTABLE no of clusers, positive integer
        self.S_m = np.zeros(self.M)                                                         ## S_m initlaized randomly to 0, array of size M 1D
        self.DELTA = np.full(self.M, self.base(self.X))
        self.CENTERS = np.zeros((self.M, self.X.shape[0]))                                  ## M x d array, CENTERS[i] is a d-dimensional point
        self.BETA = np.ones(self.M)                                                         ## beta initlaized to 1, array of size M, 1D
        self.HistoryBETA = []; self.HistoryETA = []; self.HistoryDELTA = []; self.HistoryGSI = []

    def normalize(self) -> None:
        Xmin = self.X.min(axis = 1, keepdims = True)                                        ## Xmin is size d x 1 = min as defined in the paper
        Xmax = self.X.max(axis = 1, keepdims = True)                                        ## Xmax is size d x 1 = max as defined in the paper
        denominator = Xmax - Xmin; denominator[denominator == 0] = 1.0                      ## to prevent division by 0
        self.X = (self.X - Xmin) / denominator                                              ## IMMUTABLE normalized datapoints size d x n

    def base(self, X) -> np.ndarray:
        nn = 2 * X.shape[1]
        sum_X = X.sum(axis = 0); sum_X[sum_X == 0] = 1.0                                    ## to avoid division by 0
        base = X.min(axis = 0) / sum_X                                                      ## value of delta_m with beta set = 1 as defined in paper
        return float((1 / nn) * base.sum())                                                 ## returns a real number

    def global_silhouette_score(self, y: np.ndarray, batch_size: int = 2048) -> float:
        n = self.X.shape[1]
        silhouette_scores = np.zeros(n, dtype = float)                                      ## initialize individual scores
        counts = np.bincount(y, minlength = self.M)                                         ## count of points in each cluster
        Y_oh = np.zeros((n, self.M)); Y_oh[np.arange(n), y] = 1.0                           ## one-hot encoded clusters, n x M
        X_sq = np.sum(self.X**2, axis = 0)                                                  ## precompute ||X||^2, shape (n, )
        for i in range(0, n, batch_size):                                                   ## batch processing to avoid O(n^2) space
            X_batch = self.X[:, i:i+batch_size]; B = X_batch.shape[1]                       ## extract batch
            D_sq = X_sq[i:i+batch_size, None] + X_sq[None, :] - 2 * np.dot(X_batch.T, self.X)
            D = np.sqrt(np.maximum(D_sq, 0.0))                                              ## true euclidean distances, B x n
            D[np.arange(B), np.arange(i, i+B)] = 0.0                                        ## explicitly zero out self-distances (i == j)
            sum_D = np.dot(D, Y_oh)                                                         ## B x M, sum of L2 norms to each cluster
            avg_D = np.full((B, self.M), np.inf)                                            ## average distances, B x M
            valid_c = (counts > 0)                                                          ## mask for non-empty clusters
            avg_D[:, valid_c] = sum_D[:, valid_c] / counts[valid_c]                         ## compute running averages
            batch_y = y[i:i+B]; own_count = counts[batch_y] - 1.0                           ## points in own cluster (excluding self)
            valid_own = (own_count > 0)                                                     ## mask for valid intra-cluster counts
            a = np.full(B, np.inf)                                                          ## initialize 'a'
            a[valid_own] = sum_D[valid_own, batch_y[valid_own]] / own_count[valid_own]      ## intra-cluster mean distance
            avg_D[np.arange(B), batch_y] = np.inf                                           ## exclude own cluster from 'b' candidates
            b = np.min(avg_D, axis = 1); maximum = np.maximum(b, a)                         ## compute 'b' and the denominator max(a, b)
            sil = np.zeros(B)                                                               ## initialize batch silhouette scores
            valid_sil = (maximum > 0) & ~np.isinf(maximum)                                  ## protect against division by zero or inf
            sil[valid_sil] = (b[valid_sil] - a[valid_sil]) / maximum[valid_sil]             ## compute scores for valid points
            silhouette_scores[i:i+B] = sil                                                  ## store batch results
        for m in range(self.M):
            mask_m = (y == m)
            self.S_m[m] = silhouette_scores[mask_m].mean() if np.any(mask_m) else 0.0       ## update cluster-wise average scores
        return float(silhouette_scores.mean())
        
    def mountain_train(self, batch_sz: int = 2048) -> None:                                 ## step 2 to 6 as defined in paper
        X_copy = self.X.copy()                                                              ## make a copy cuz we will be deleting columns in mth cluster
        for m in range(self.M):
            if X_copy.shape[1] == 0: break                                                  ## means there are no more clusters at all 
            self.DELTA[m] = self.BETA[m] * self.base(X_copy)                                ## update DELTA as defined in paper
            X_sq = np.sum(X_copy**2, axis = 0)                                              ## X_sq[i] = l2norm(Xi), this is size 1 x n
            prm = np.zeros(X_copy.shape[1])                                                 ## initialize potentials array for the copied data
            for i in range(0, X_copy.shape[1], batch_sz):                                   ## batch processing to avoid O(n^2) space
                X_batch = X_copy[:, i:i+batch_sz]                                           ## extract batch
                l2norm_batch = X_sq[i:i+batch_sz, None] + X_sq[None, :] - 2 * np.dot(X_batch.T, X_copy)
                kernel_batch = np.exp(- l2norm_batch / (self.DELTA[m]**2))                  ## Kernel as defined in the paper, batch_size x n
                prm[i:i+batch_sz] = np.sum(kernel_batch, axis = 1)                          ## accumulate potentials for batch
            mountain_peak = np.argmax(prm)                                                  ## which has the highest potential to be made the mth peak
            self.CENTERS[m] = X_copy[:, mountain_peak]                                      ## update the mth center to its real value
            center_sq = np.sum(self.CENTERS[m]**2)                                          ## l2norm(mth center)
            dists_to_center = X_sq + center_sq - 2 * np.dot(self.CENTERS[m], X_copy)        ## size 1 x n, ith entry being ||Xi - center_m||**2
            keep_mask = (dists_to_center > (self.DELTA[m]**2))                              ## the data points whih are far from peak as defined in paper
            X_copy = X_copy[:, keep_mask]                                                   ## delete the points which are in the mth cluster
            print(f"  [mountain] cluster {m+1}/{self.M} done", end = '\r')

    def SOC_train(self, iterations: int = 10, batch_size: int = 2048) -> None:
        GSI_best = -1.0
        best_centers = None
        for i in tqdm(range(iterations), desc = "Training SOC"):
            print(f"Training Iteration {i+1}/{iterations}...", end = '\r')
            self.mountain_train(batch_sz = batch_size)
            y = self.predict(self.X)
            gsi = self.global_silhouette_score(y, batch_size)                               ## evaluate GSI after mountian_train() whcich makes self.S_m
            if gsi >= GSI_best:
                GSI_best = gsi;
                best_centers = self.CENTERS.copy()
            delta_safe = self.DELTA + np.arange(self.M) * 1e-12
            poly = lagrange(delta_safe, self.S_m)
            if poly.order > 0:
                roots = (poly - 1.0).roots
                real_roots = roots[np.isclose(roots.imag, 0)].real
                valid_roots = real_roots[(real_roots > 0) & (real_roots <= 0.1666)]
                eta = valid_roots[np.argmin(np.abs(poly(valid_roots) - 1.0))] if valid_roots.size > 0 else 0.1666
            else: eta = 0.1666
            self.BETA = eta / self.DELTA                                                     ## find eta, update BETA as per paper
            self.store_data(eta, gsi)
        self.CENTERS = best_centers

    def train(self, iterations: int = 10, batch_size: int = 2048) -> None:                   ## by defaiult, run it 10 times
        self.SOC_train(iterations, batch_size)

    def predict(self, X, batch_size: int = 2048) -> np.ndarray:
        n = X.shape[1]
        predictions = np.zeros(n, dtype = int)
        C_sq = np.sum(self.CENTERS**2, axis = 1)                                             ## Precompute ||C||^2, shape (M, )
        for i in range(0, n, batch_size): 
            X_batch = X[:, i:i+batch_size]                                                   ## Extract batch, shape (d, B)
            X_batch_sq = np.sum(X_batch**2, axis = 0)                                        ## ||X||^2 for batch, shape (B, )
            dists_sq = X_batch_sq[:, None] + C_sq[None, :] - 2 * np.dot(X_batch.T, self.CENTERS.T)
            predictions[i:i+batch_size] = np.argmin(dists_sq, axis = 1)                      ## Store assignments for batch
        return predictions

    def store_data(self, eta: float, GSI: float) -> None:
        self.HistoryGSI.append(GSI)
        self.HistoryBETA.append(self.BETA)
        self.HistoryDELTA.append(self.DELTA)
        self.HistoryETA.append(eta)

    def print_charts(self) -> None:
        print("Iter   | Beta                                   | Delta                                  | Eta     | GSI     ")
        print("-------------------------------------------------------------------------------------------------------------")
        for i in range(len(self.HistoryGSI)):
            beta_str = str([round(x, 2) for x in self.HistoryBETA[i]])
            delta_str = str([round(x, 2) for x in self.HistoryDELTA[i]])
            eta_val = round(self.HistoryETA[i], 2)
            gsi_val = round(self.HistoryGSI[i], 2)
            print(f"{i+1:<6} | {beta_str:<35} | {delta_str:<35} | {eta_val:<8} | {gsi_val:<8}")

    def get_history(self) -> dict:
        return { "BETA" : self.HistoryBETA, "ETA" : self.HistoryETA, "DELTA" : self.HistoryDELTA, "GSI" : self.HistoryGSI }

    def plot_graphs(self) -> None:
        import matplotlib.pyplot as plt                                                 
        x = np.arange(1, len(self.HistoryGSI) + 1)                                            ## x-axis: 1 to number of iterations
        y = np.array(self.HistoryGSI)                                                         ## y-axis: GSI history array
        plt.figure(figsize = (8, 6))                                                          ## setup canvas
        plt.plot(x, y, color = 'red', linestyle = '--', marker = 's', mfc = 'green')          ## plot matching the paper's style (Fig 1)
        best_iter = np.argmax(y)                                                              ## find index of the best GSI
        plt.plot(x[best_iter], y[best_iter], marker = 's', color = 'black')                   ## highlight the optimal iteration point
        annot_text = f"X: {x[best_iter]}\nY: {y[best_iter]:.4f}"                              ## format annotation text 
        plt.annotate(annot_text, xy = (x[best_iter], y[best_iter]),                           ## add annotation box near the best point
                     xytext = (10, -30), textcoords = 'offset points', 
                     bbox = dict(boxstyle = "square,pad = 0.3", fc = "lightyellow", ec = "silver"))
        plt.xlabel('Number of iterations'); plt.ylabel('Global Silhouette Index (GSI)')       ## set axis labels
        plt.grid(True)                                                                        ## enable background grid
        plt.show()                                                                            ## display the plot
        