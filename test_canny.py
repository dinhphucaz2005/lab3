import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

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

def rgb_to_gray(img):
    if len(img.shape) == 2:
        return img
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.uint8)

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

def canny_edge_detection(image, blur_size=5, blur_sigma=1.0, low_ratio=0.05, high_ratio=0.09, high_threshold=None, low_threshold=None):
    kernel = gaussian_kernel(blur_size, sigma=blur_sigma)
    img_smoothed = custom_convolve(image, kernel)
    
    G, theta = sobel_filters(img_smoothed)
    nms = non_max_suppression(G, theta)
    dt, weak, strong = double_threshold(nms, low_ratio, high_ratio, high_threshold=high_threshold, low_threshold=low_threshold)
    out = hysteresis(dt, weak, strong)
    return out

def otsu_threshold(img):
    flat = img.flatten()
    hist, bins = np.histogram(flat, bins=256, range=(0, 256))
    total = flat.size
    sum_all = np.dot(np.arange(256), hist)
    
    sumB, wB = 0, 0
    maximum = 0.0
    threshold1 = 0.0
    
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
            
        sumB += float(i * hist[i])
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        varBetween = wB * wF * (mB - mF) ** 2
        
        if varBetween > maximum:
            maximum = varBetween
            threshold1 = i
    return threshold1

def calculate_iou(pred_edges, true_edges):
    pred = (pred_edges > 0).astype(bool)
    true = (true_edges > 0).astype(bool)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 0.0
    return intersection / union

def run_test():
    print("--- TESTING CANNY PIPELINE (EXERCISE 1) ---")
    if os.path.exists('image.png'):
        print("Loading image.png...")
        color_img = cv2.imread('image.png', cv2.IMREAD_COLOR)
        img = rgb_to_gray(color_img)
    else:
        print("image.png not found. Creating random image...")
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)

    # Test Canny with manual thresholding
    print("Running Canny Edge Detection (Manual)...")
    canny_manual = canny_edge_detection(img, high_threshold=80, low_threshold=30)
    
    # Test Otsu thresholding
    print("Calculating Otsu threshold...")
    kernel = gaussian_kernel(5, sigma=1.0)
    img_smoothed = custom_convolve(img, kernel)
    G, _ = sobel_filters(img_smoothed)
    tau_high_otsu = otsu_threshold(G)
    tau_low_otsu = 0.5 * tau_high_otsu
    print(f"Otsu High: {tau_high_otsu}, Low: {tau_low_otsu}")
    
    # Test Canny with Otsu
    print("Running Canny Edge Detection (Otsu)...")
    canny_otsu = canny_edge_detection(img, high_threshold=tau_high_otsu, low_threshold=tau_low_otsu)
    
    # Test against opencv
    print("Comparing with cv2.Canny...")
    cv2_edges = cv2.Canny(img, 30, 80)
    iou_manual = calculate_iou(canny_manual, cv2_edges)
    iou_otsu = calculate_iou(canny_otsu, cv2_edges)
    print(f"IoU (Manual) vs OpenCV: {iou_manual:.4f}")
    print(f"IoU (Otsu) vs OpenCV: {iou_otsu:.4f}")
    
    # Assert output properties
    assert canny_manual.shape == img.shape, "Shape mismatch"
    assert canny_otsu.shape == img.shape, "Shape mismatch"
    assert np.all(np.isin(canny_manual, [0, 255])), "Canny output should only be 0 or 255"
    assert np.all(np.isin(canny_otsu, [0, 255])), "Canny output should only be 0 or 255"
    
    # Save test visualization to png
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Grayscale")
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title("Manual Canny (30, 80)")
    plt.imshow(canny_manual, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title(f"Otsu Canny ({tau_low_otsu:.1f}, {tau_high_otsu:.1f})")
    plt.imshow(canny_otsu, cmap='gray')
    plt.tight_layout()
    plt.savefig('test_canny_result.png')
    print("Saved visualization to test_canny_result.png")
    print("--- CANNY PIPELINE TESTS PASSED SUCCESSFULY ---")

if __name__ == '__main__':
    run_test()
