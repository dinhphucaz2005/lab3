import numpy as np
import matplotlib.pyplot as plt

def ransac(points, fit_fn, distance_fn, s, max_iters=100, threshold=1.0, min_inliers=10):
    best_model = None
    best_inliers_mask = None
    max_inlier_count = -1
    
    N = len(points)
    if N < s:
        return None, None
        
    for i in range(max_iters):
        idx = np.random.choice(N, s, replace=False)
        sample = points[idx]
        
        model = fit_fn(sample)
        if model is None:
            continue
            
        distances = distance_fn(model, points)
        inliers_mask = distances < threshold
        inlier_count = np.sum(inliers_mask)
        
        if inlier_count > max_inlier_count and inlier_count >= min_inliers:
            max_inlier_count = inlier_count
            best_inliers_mask = inliers_mask
            best_model = fit_fn(points[inliers_mask])
            
    return best_model, best_inliers_mask

# Line functions
def fit_line(pts):
    if len(pts) < 2:
        return None
    if len(pts) == 2:
        p1, p2 = pts[0], pts[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0 and dy == 0:
            return None
        a, b = -dy, dx
        norm = np.hypot(a, b)
        a, b = a / norm, b / norm
        c = -(a * p1[0] + b * p1[1])
        return (a, b, c)
    else:
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        cov = np.dot(pts_centered.T, pts_centered)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        a, b = eigenvectors[:, 0]
        c = -(a * centroid[0] + b * centroid[1])
        return (a, b, c)

def distance_line(model, pts):
    a, b, c = model
    return np.abs(a * pts[:, 0] + b * pts[:, 1] + c)

# Circle functions
def fit_circle(pts):
    if len(pts) < 3:
        return None
    X = pts[:, 0]
    Y = pts[:, 1]
    Z = X**2 + Y**2
    M = np.column_stack((X, Y, np.ones_like(X)))
    
    try:
        v, _, _, _ = np.linalg.lstsq(M, Z, rcond=None)
        a, b, c = v[0], v[1], v[2]
        x_c = a / 2.0
        y_c = b / 2.0
        R_sq = c + x_c**2 + y_c**2
        if R_sq <= 0:
            return None
        return (x_c, y_c, np.sqrt(R_sq))
    except np.linalg.LinAlgError:
        return None

def distance_circle(model, pts):
    x_c, y_c, R = model
    dist_to_center = np.hypot(pts[:, 0] - x_c, pts[:, 1] - y_c)
    return np.abs(dist_to_center - R)

# Data generators
def generate_line_data(num_pts=200, outlier_ratio=0.2, sigma=0.2):
    num_outliers = int(num_pts * outlier_ratio)
    num_inliers = num_pts - num_outliers
    
    x_in = np.linspace(-10, 10, num_inliers)
    y_in = 1.5 * x_in - 2.0 + np.random.normal(0, sigma, num_inliers)
    inliers = np.column_stack((x_in, y_in))
    
    x_out = np.random.uniform(-10, 10, num_outliers)
    y_out = np.random.uniform(-17, 13, num_outliers)
    outliers = np.column_stack((x_out, y_out))
    
    pts = np.vstack((inliers, outliers))
    np.random.shuffle(pts)
    return pts

def generate_circle_data(num_pts=200, outlier_ratio=0.2, sigma=0.2):
    num_outliers = int(num_pts * outlier_ratio)
    num_inliers = num_pts - num_outliers
    
    x_c, y_c, R = 1.0, 2.0, 5.0
    theta = np.linspace(0, 2*np.pi, num_inliers)
    x_in = x_c + R * np.cos(theta) + np.random.normal(0, sigma, num_inliers)
    y_in = y_c + R * np.sin(theta) + np.random.normal(0, sigma, num_inliers)
    inliers = np.column_stack((x_in, y_in))
    
    x_out = np.random.uniform(-7, 9, num_outliers)
    y_out = np.random.uniform(-6, 10, num_outliers)
    outliers = np.column_stack((x_out, y_out))
    
    pts = np.vstack((inliers, outliers))
    np.random.shuffle(pts)
    return pts

def run_test():
    print("--- TESTING RANSAC PIPELINE (EXERCISE 3) ---")
    
    # 1. Line Test
    print("Testing Line fitting with 30% outliers...")
    pts_line = generate_line_data(num_pts=200, outlier_ratio=0.3, sigma=0.2)
    model_l, inliers_l = ransac(pts_line, fit_line, distance_line, s=2, max_iters=100, threshold=0.6, min_inliers=10)
    
    assert model_l is not None, "Failed to fit line model"
    a, b, c = model_l
    # Verify line model parameters are finite
    assert np.isfinite(a) and np.isfinite(b) and np.isfinite(c), "Invalid line parameters"
    print(f"Fitted Line parameters (a, b, c): ({a:.4f}, {b:.4f}, {c:.4f})")
    
    # 2. Circle Test
    print("Testing Circle fitting with 30% outliers...")
    pts_circle = generate_circle_data(num_pts=200, outlier_ratio=0.3, sigma=0.2)
    model_c, inliers_c = ransac(pts_circle, fit_circle, distance_circle, s=3, max_iters=150, threshold=0.6, min_inliers=10)
    
    assert model_c is not None, "Failed to fit circle model"
    xc, yc, R = model_c
    assert np.isfinite(xc) and np.isfinite(yc) and np.isfinite(R), "Invalid circle parameters"
    print(f"Fitted Circle parameters (xc, yc, R): ({xc:.4f}, {yc:.4f}, {R:.4f})")
    
    # Plot and save
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Line plot
    axes[0].scatter(pts_line[:, 0], pts_line[:, 1], color='gray', alpha=0.5, label='Points')
    axes[0].scatter(pts_line[inliers_l, 0], pts_line[inliers_l, 1], color='blue', alpha=0.7, label='Inliers')
    x_vals = np.array([-10, 10])
    y_vals = -(a * x_vals + c) / b
    axes[0].plot(x_vals, y_vals, color='red', linewidth=2, label='Fitted Line')
    axes[0].set_title("RANSAC Line Fitting")
    axes[0].legend()
    
    # Circle plot
    axes[1].scatter(pts_circle[:, 0], pts_circle[:, 1], color='gray', alpha=0.5, label='Points')
    axes[1].scatter(pts_circle[inliers_c, 0], pts_circle[inliers_c, 1], color='green', alpha=0.7, label='Inliers')
    t_draw = np.linspace(0, 2*np.pi, 200)
    axes[1].plot(xc + R * np.cos(t_draw), yc + R * np.sin(t_draw), color='red', linewidth=2, label='Fitted Circle')
    axes[1].set_title("RANSAC Circle Fitting")
    axes[1].legend()
    
    plt.savefig('test_ransac_result.png')
    print("Saved visualization to test_ransac_result.png")
    print("--- RANSAC PIPELINE TESTS PASSED SUCCESSFULY ---")

if __name__ == '__main__':
    run_test()
