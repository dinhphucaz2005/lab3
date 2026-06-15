import numpy as np
import cv2
import matplotlib.pyplot as plt

def custom_convolve(image, kernel):
    kernel_flipped = np.flip(kernel)
    kh, kw = kernel.shape
    img_h, img_w = image.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')
    padded = np.ascontiguousarray(padded)
    
    shape = (img_h, img_w, kh, kw)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    patches = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    
    return np.tensordot(patches, kernel_flipped, axes=((2, 3), (0, 1)))

def gaussian_kernel(size, sigma=1.0):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = custom_convolve(img.astype(np.float32), Kx)
    Iy = custom_convolve(img.astype(np.float32), Ky)
    
    G = np.hypot(Ix, Iy)
    g_max = G.max()
    if g_max > 0:
        G = G / g_max * 255
    theta = np.arctan2(Iy, Ix)
    return G, theta

def non_max_suppression(img, D):
    M, N = img.shape
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    img_pad = np.pad(img, ((1,1), (1,1)), 'constant')
    
    mask_0 = ((angle >= 0) & (angle < 22.5)) | ((angle >= 157.5) & (angle <= 180))
    mask_45 = (angle >= 22.5) & (angle < 67.5)
    mask_90 = (angle >= 67.5) & (angle < 112.5)
    mask_135 = (angle >= 112.5) & (angle < 157.5)

    q0 = img_pad[1:M+1, 2:N+2]
    r0 = img_pad[1:M+1, 0:N]
    q45 = img_pad[0:M, 2:N+2]
    r45 = img_pad[2:M+2, 0:N]
    q90 = img_pad[0:M, 1:N+1]
    r90 = img_pad[2:M+2, 1:N+1]
    q135 = img_pad[0:M, 0:N]
    r135 = img_pad[2:M+2, 2:N+2]

    q = np.zeros_like(img)
    r = np.zeros_like(img)
    
    q[mask_0] = q0[mask_0]
    r[mask_0] = r0[mask_0]
    q[mask_45] = q45[mask_45]
    r[mask_45] = r45[mask_45]
    q[mask_90] = q90[mask_90]
    r[mask_90] = r90[mask_90]
    q[mask_135] = q135[mask_135]
    r[mask_135] = r135[mask_135]

    suppressed = (img >= q) & (img >= r)
    Z = np.where(suppressed, img, 0.0)
    return Z

def double_threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09, high_threshold=None, low_threshold=None):
    if high_threshold is None:
        highThreshold = img.max() * highThresholdRatio
    else:
        highThreshold = high_threshold
    if low_threshold is None:
        lowThreshold = highThreshold * lowThresholdRatio
    else:
        lowThreshold = low_threshold
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong

def hysteresis(img, weak, strong=255):
    M, N = img.shape
    img_out = np.copy(img)
    strong_i, strong_j = np.where(img_out == strong)
    stack = list(zip(strong_i, strong_j))
    
    while stack:
        i, j = stack.pop()
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ni, nj = i + dx, j + dy
                if 0 <= ni < M and 0 <= nj < N:
                    if img_out[ni, nj] == weak:
                        img_out[ni, nj] = strong
                        stack.append((ni, nj))
    img_out[img_out == weak] = 0
    return img_out

def canny_edge_detection(image, blur_size=5, blur_sigma=1.0, low_threshold=30, high_threshold=80):
    kernel = gaussian_kernel(blur_size, sigma=blur_sigma)
    img_smoothed = custom_convolve(image, kernel)
    
    G, theta = sobel_filters(img_smoothed)
    nms = non_max_suppression(G, theta)
    dt, weak, strong = double_threshold(nms, high_threshold=high_threshold, low_threshold=low_threshold)
    out = hysteresis(dt, weak, strong)
    return out

def hough_vote_directed(edge_img, theta_bins, r_bins, phi_img, window_size_rad=None):
    H, W = edge_img.shape
    y_idxs, x_idxs = np.nonzero(edge_img)
    num_r = len(r_bins)
    num_theta = len(theta_bins)
    accumulator = np.zeros((num_r, num_theta), dtype=np.int32)
    
    if len(y_idxs) == 0:
        return accumulator
        
    cos_t = np.cos(theta_bins)
    sin_t = np.sin(theta_bins)
    
    r_min = r_bins[0]
    r_step = r_bins[1] - r_bins[0] if len(r_bins) > 1 else 1.0
    
    r_vals = np.outer(x_idxs, cos_t) + np.outer(y_idxs, sin_t)
    r_idxs = np.round((r_vals - r_min) / r_step).astype(np.int32)
    
    valid_r = (r_idxs >= 0) & (r_idxs < num_r)
    
    if window_size_rad is not None:
        phi_vals = phi_img[y_idxs, x_idxs][:, np.newaxis]
        diff = np.abs(theta_bins - phi_vals)
        diff = np.minimum(diff, np.pi - diff)
        valid_theta = diff <= window_size_rad
        mask = valid_r & valid_theta
    else:
        mask = valid_r
        
    pixel_idxs, theta_idxs = np.where(mask)
    r_to_acc = r_idxs[pixel_idxs, theta_idxs]
    
    np.add.at(accumulator, (r_to_acc, theta_idxs), 1)
    return accumulator

def find_peaks_nms(accumulator, num_peaks, neighborhood_size=9, threshold=10):
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(accumulator, size=neighborhood_size)
    maxima = (accumulator == local_max) & (accumulator > threshold)
    
    r_indices, t_indices = np.nonzero(maxima)
    votes = accumulator[r_indices, t_indices]
    
    sort_idx = np.argsort(votes)[::-1]
    r_indices = r_indices[sort_idx][:num_peaks]
    t_indices = t_indices[sort_idx][:num_peaks]
    return r_indices, t_indices

def coarse_to_fine_hough(edge_img, phi_img, num_peaks, window_size_rad=None):
    H, W = edge_img.shape
    theta_coarse = np.linspace(-np.pi/2, np.pi/2, 90, endpoint=False)
    r_max = np.sqrt(H**2 + W**2)
    r_coarse = np.linspace(-r_max, r_max, int(r_max / 2))
    
    acc_coarse = hough_vote_directed(edge_img, theta_coarse, r_coarse, phi_img, window_size_rad)
    r_coarse_idxs, t_coarse_idxs = find_peaks_nms(acc_coarse, num_peaks, neighborhood_size=9, threshold=10)
    
    refined_lines = []
    t_step_coarse = theta_coarse[1] - theta_coarse[0] if len(theta_coarse) > 1 else 1.0
    r_step_coarse = r_coarse[1] - r_coarse[0] if len(r_coarse) > 1 else 1.0
    
    for ri, ti in zip(r_coarse_idxs, t_coarse_idxs):
        t_c = theta_coarse[ti]
        r_c = r_coarse[ri]
        
        theta_fine = np.linspace(t_c - t_step_coarse, t_c + t_step_coarse, 21)
        r_fine = np.linspace(r_c - r_step_coarse, r_c + r_step_coarse, 21)
        
        acc_fine = hough_vote_directed(edge_img, theta_fine, r_fine, phi_img, window_size_rad)
        best_r_idx, best_t_idx = np.unravel_index(np.argmax(acc_fine), acc_fine.shape)
        refined_lines.append((r_fine[best_r_idx], theta_fine[best_t_idx]))
    return refined_lines

def generate_chessboard(size=240, grid_size=8, noise_std=15):
    img = np.zeros((size, size), dtype=np.uint8)
    cell_size = size // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 0:
                img[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] = 255
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def evaluate_lines(detected_lines, gt_lines, r_tol=6, theta_tol=np.radians(6)):
    tp = 0
    matched_gt = set()
    for r_det, t_det in detected_lines:
        for idx, (r_gt, t_gt) in enumerate(gt_lines):
            if idx in matched_gt:
                continue
            r_diff = np.abs(r_det - r_gt)
            t_diff = np.abs(t_det - t_gt)
            t_diff = np.minimum(t_diff, np.pi - t_diff)
            
            if r_diff <= r_tol and t_diff <= theta_tol:
                tp += 1
                matched_gt.add(idx)
                break
    fp = len(detected_lines) - tp
    fn = len(gt_lines) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def run_test():
    print("--- TESTING HOUGH PIPELINE (EXERCISE 2) ---")
    print("Generating synthetic chessboard image...")
    chessboard = generate_chessboard(size=240, grid_size=8, noise_std=15)
    
    # Run Sobel filters for gradient directions
    kernel = gaussian_kernel(5, sigma=1.0)
    chessboard_smoothed = custom_convolve(chessboard, kernel)
    G, theta = sobel_filters(chessboard_smoothed)
    
    phi_img = theta.copy()
    phi_img[phi_img > np.pi/2] -= np.pi
    phi_img[phi_img < -np.pi/2] += np.pi
    
    edge_img = canny_edge_detection(chessboard, low_threshold=30, high_threshold=80)
    
    # Test Hough with k = 5
    print("Running Coarse-to-Fine Hough with k = 5...")
    detected_lines = coarse_to_fine_hough(edge_img, phi_img, num_peaks=5, window_size_rad=None)
    print(f"Detected {len(detected_lines)} lines.")
    
    # Ground truth coords
    cell_size = 30
    grid_coords = [i * cell_size for i in range(1, 8)]
    gt_lines = []
    for x in grid_coords:
        gt_lines.append((float(x), 0.0))
    for y in grid_coords:
        gt_lines.append((float(y), np.pi/2))
        
    p, r, f1 = evaluate_lines(detected_lines, gt_lines)
    print(f"k = 5 performance evaluation: Precision = {p:.4f}, Recall = {r:.4f}, F1-score = {f1:.4f}")
    
    assert len(detected_lines) <= 5, "Should return at most 5 lines"
    assert all(isinstance(line, tuple) and len(line) == 2 for line in detected_lines), "Output format mismatch"
    
    # Save visualization to png
    disp_img = cv2.cvtColor(chessboard, cv2.COLOR_GRAY2BGR)
    for r_val, t_val in detected_lines:
        a = np.cos(t_val)
        b = np.sin(t_val)
        x0 = a * r_val
        y0 = b * r_val
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(disp_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
    plt.figure(figsize=(6, 6))
    plt.title("Hough lines with k=5")
    plt.imshow(disp_img)
    plt.savefig('test_hough_result.png')
    print("Saved visualization to test_hough_result.png")
    print("--- HOUGH PIPELINE TESTS PASSED SUCCESSFULY ---")

if __name__ == '__main__':
    run_test()
