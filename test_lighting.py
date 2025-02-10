import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def estimate_light_direction(image_path, data, type, number, num_patches=8):
    """Estimate light direction using occluding boundaries."""
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def fit_quadratic_curve(points):
        """Fit quadratic curve to boundary points."""
        x = points[:, 0]
        y = points[:, 1]
        # x_centered = x - x.mean()
        coeffs = np.polyfit(x, y, 2)
        return coeffs
    
    def compute_surface_normals(coeffs, x_points):
        """Compute 2D surface normals from quadratic fit."""
        # Derivative of quadratic ax^2 + bx + c is 2ax + b
        derivative = np.polyder(coeffs)
        dy_dx = np.polyval(derivative, x_points)
        # Normal is perpendicular to tangent [-dy/dx, 1]
        normals = np.column_stack((-dy_dx, np.ones_like(dy_dx)))
        # Normalize
        norms = np.linalg.norm(normals, axis=1)[:, np.newaxis]
        normals /= norms
        return normals
    
    def measure_intensities(points, normals, offset=1):
        """Measure intensities near boundary points."""
        intensities = []
        for point, normal in zip(points, normals):
            x, y = point.astype(int)
            dx, dy = normal
            # Sample point slightly away from boundary
            sample_x = int(x - offset * dx)
            sample_y = int(y - offset * dy)
            
            if (0 <= sample_x < gray_image.shape[1] and 
                0 <= sample_y < gray_image.shape[0]):
                intensities.append(gray_image[sample_y, sample_x])
        return np.array(intensities)
    
    def estimate_patch_light_direction(points, intensities, normals):
        """Estimate light direction for a single patch."""
        # if len(points) < 3:
        #     return None
            
        # Set up least squares system
        M = np.column_stack((normals, np.ones(len(normals))))
        try:
            # Solve for [Lx, Ly, A] where A is ambient term
            v = np.linalg.lstsq(M, intensities, rcond=None)[0]
            return v[:2]  # Return only light direction components
        except np.linalg.LinAlgError:
            return None
    
    def process_boundary(boundary_points):
        """Process boundary points to estimate light direction."""
        # Split boundary into patches
        n_points = len(boundary_points)
        print(n_points)
        patch_size = n_points // num_patches
        light_directions = []
        print("NumPatches: ", num_patches)
        for i in range(num_patches):
            start_idx = i * patch_size
            end_idx = start_idx + patch_size
            
            patch_points = boundary_points[start_idx:end_idx]
            print("Patchpts: ", patch_points)
            # if len(patch_points) < 3:
            #     continue
                
            # Fit quadratic to patch points
            coeffs = fit_quadratic_curve(patch_points)
            print("Coeffs: ", coeffs)
            x_points = patch_points[:, 0]
            
            # Compute surface normals
            normals = compute_surface_normals(coeffs, x_points)
            print("normals: ", normals)
            # Measure intensities
            intensities = measure_intensities(patch_points, normals)
            print("intensities: ", intensities)
            # Estimate light direction for patch
            direction = estimate_patch_light_direction(
                patch_points, intensities, normals)
            print("direction: ", direction)
            if direction is not None:
                light_directions.append(direction)
        
        return np.array(light_directions)
    
    def visualize_results(boundary_points, light_directions, avg_direction):
        """Visualize results with boundary and light directions."""
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(f'output_lg/{data}/orig_{type}_{number}.jpg'.format(type, number), bbox_inches='tight')
        
        
        # Light direction
        # plt.title('Light Direction')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Plot average direction
        center = np.mean(boundary_points, axis=0)
        length = min(image.shape[:2]) // 4
        angle = np.arctan2(avg_direction[1], avg_direction[0])
        dx = length * np.cos(angle)
        dy = length * np.sin(angle)
        
        plt.arrow(center[0], center[1], dx, dy, 
                 color='yellow', width=2, head_width=10)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'output_lg/{data}/lgcue_{type}_{number}.jpg'.format(type, number), bbox_inches='tight')
        plt.show()
    
    edges = cv2.Canny(gray_image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_NONE)
    
    # Use largest contour as boundary
    if not contours:
        raise ValueError("No boundaries detected")
    
    boundary_points = contours[0].squeeze()
    light_directions = process_boundary(boundary_points)
    print("hi")
    print("LenofLight: ", len(light_directions))
    if len(light_directions) == 0:
        raise ValueError("Could not estimate light direction")
    print("hi2")
    # Compute average direction
    avg_direction = np.mean(light_directions, axis=0)
    avg_direction /= np.linalg.norm(avg_direction)
    print("hi")
    # Visualize results
    visualize_results(boundary_points, light_directions, avg_direction)
    
    # Convert to degrees
    angle_degrees = np.degrees(np.arctan2(avg_direction[1], avg_direction[0]))
    return angle_degrees
#dsd


#DeepFloyd Outdoor: , 249, 625, 1121, 500, 250
#ShadowDir: 624, 750

# Shadow_dir:250, 996
# Kandinsky_outdoor: 496, 500(sha), , 1123,, 624
# Example usage
number = 2247
type = 'real'
data = 'Deepfloyd_Outdoor'
image_path = f'dataset/{data}/val/{type}/{number}.jpg'.format(data, type, number)
try:
    angle = estimate_light_direction(image_path, data, type, number)
    print(f"Estimated light source direction: {angle:.2f} degrees")
except Exception as e:
    print(f"Error: {str(e)}")

# Example usage
type = 'gen'
image_path = f'dataset/{data}/val/{type}/{number}.jpg'.format(data, type, number)
try:
    angle = estimate_light_direction(image_path, data, type, number)
    print(f"Estimated light source direction: {angle:.2f} degrees")
except Exception as e:
    print(f"Error: {str(e)}")

