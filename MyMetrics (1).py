import numpy as np                     
























def global_silhouette_score(X: np.ndarray, y: np.ndarray, M: int, batch_size: int = 2048) -> float:
        n = X.shape[1]
        silhouette_scores = np.zeros(n, dtype = float)                                      ## initialize individual scores
        counts = np.bincount(y, minlength = M)                                              ## count of points in each cluster
        Y_oh = np.zeros((n, M)); Y_oh[np.arange(n), y] = 1.0                                ## one-hot encoded clusters, n x M
        X_sq = np.sum(X**2, axis = 0)                                                       ## precompute ||X||^2, shape (n, )
        for i in range(0, n, batch_size):                                                   ## batch processing to avoid O(n^2) space
            X_batch = X[:, i:i+batch_size]; B = X_batch.shape[1]                            ## extract batch
            D_sq = X_sq[i:i+batch_size, None] + X_sq[None, :] - 2 * np.dot(X_batch.T, X)
            D = np.sqrt(np.maximum(D_sq, 0.0))                                              ## true euclidean distances, B x n
            D[np.arange(B), np.arange(i, i+B)] = 0.0                                        ## explicitly zero out self-distances (i == j)
            sum_D = np.dot(D, Y_oh)                                                         ## B x M, sum of L2 norms to each cluster
            avg_D = np.full((B, M), np.inf)                                                 ## average distances, B x M
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
        return float(silhouette_scores.mean())

def partition_index(X: np.ndarray, y: np.ndarray, M: int, batch_size: int = 2048) -> float:
    d, n = X.shape
    centers = np.zeros((d, M))
    for m in range(M):
        mask_m = (y == m)
        if np.sum(mask_m) > 0:
            centers[:, m] = np.mean(X[:, mask_m], axis=1)                           ## compute empirical centroid for cluster m
    C_sq = np.sum(centers**2, axis=0)                                               ## ||c_m||^2, size M
    inter_dist_sq = C_sq[:, None] + C_sq[None, :] - 2 * np.dot(centers.T, centers)  ## M x M matrix of ||c_k - c_m||^2
    np.fill_diagonal(inter_dist_sq, 0)                                              ## distance to self is 0
    PI = 0.0
    for m in range(M):
        mask_m = (y == m); Nm = np.sum(mask_m)                                      ## mask_m is (n, ) boolean, Nm is cardinality
        if Nm == 0: continue                                                        ## continue if cluster is empty
        idx_m = np.where(mask_m)[0]                                                 ## extract indices of points in mth cluster
        intra_sum_sq = 0.0
        for i in range(0, len(idx_m), batch_size):                                  ## batch processing
            batch_idx = idx_m[i:i+batch_size]; X_batch = X[:, batch_idx]            ## extract batch, shape (d, B)
            dists_sq = np.sum((X_batch - centers[:, m:m+1])**2, axis=0)             ## squared distances to center m
            intra_sum_sq += np.sum(dists_sq)                                        ## accumulate numerator portion
        sum_inter_sq = np.sum(inter_dist_sq[m, :])                                  ## sum of ||c_k - c_m||^2 for all k
        if sum_inter_sq > 0:
            PI += intra_sum_sq / (Nm * sum_inter_sq)                                ## accumulate PI as per equation (18)
    return float(PI)

def separation_index(X: np.ndarray, y: np.ndarray, M: int, batch_size: int = 2048) -> float:
    d, n = X.shape
    centers = np.zeros((d, M))
    for m in range(M):
        mask_m = (y == m)
        if np.sum(mask_m) > 0:
            centers[:, m] = np.mean(X[:, mask_m], axis=1)                           ## compute empirical centroid for cluster m
    C_sq = np.sum(centers**2, axis=0)                                               ## ||c_m||^2, size M
    inter_dist_sq = C_sq[:, None] + C_sq[None, :] - 2 * np.dot(centers.T, centers)  ## M x M matrix of ||c_k - c_m||^2
    np.fill_diagonal(inter_dist_sq, np.inf)                                         ## ignore self distance for minimum calculation
    min_inter_sq = np.min(inter_dist_sq)                                            ## min_{k, m} ||c_k - c_m||^2
    total_intra_sum_sq = 0.0
    for m in range(M):
        mask_m = (y == m)
        if np.sum(mask_m) == 0: continue
        idx_m = np.where(mask_m)[0]                                                 ## extract indices of points in mth cluster
        for i in range(0, len(idx_m), batch_size):                                  ## batch processing
            batch_idx = idx_m[i:i+batch_size]; X_batch = X[:, batch_idx]            ## extract batch, shape (d, B)
            dists_sq = np.sum((X_batch - centers[:, m:m+1])**2, axis=0)             ## squared distances to center m
            total_intra_sum_sq += np.sum(dists_sq)                                  ## accumulate total intracluster squared distance
    if min_inter_sq == 0 or min_inter_sq == np.inf:
        return float('inf')                                                         ## edge case handling to avoid division by zero
    SI = total_intra_sum_sq / (n * min_inter_sq)                                    ## SI as per equation (19)
    return float(SI)