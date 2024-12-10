import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
from tkinter import filedialog, messagebox
import tkinter as tk

class ImageCartoonifier:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Advanced Image Cartoonifier")
        self.window.geometry("1200x800")
        
        self.original_image = None
        self.cartoon_image = None
        
        self.setup_gui()

    def reduce_colors(self, img, n_colors=8):
        # Increase image quality
        img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter for smooth color regions
        smooth = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Convert to float32
        Z = smooth.reshape((-1, 3))
        Z = np.float32(Z)
        
        # Define criteria and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        _, labels, centers = cv2.kmeans(Z, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        # Convert back to uint8 and reshape
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        reduced = res.reshape((img.shape))
        
        # Resize back to original size
        reduced = cv2.resize(reduced, (self.original_image.shape[1], self.original_image.shape[0]), 
                           interpolation=cv2.INTER_LANCZOS4)
        
        return reduced

    def smooth_regions(self, img):
        # Multiple bilateral filtering passes
        smooth = cv2.bilateralFilter(img, 9, 75, 75)
        smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
        return smooth

    def create_edge_mask(self, img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Create edge mask
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 9, 9
        )
        
        # Clean up edges
        edges = cv2.medianBlur(edges, 3)
        
        # Dilate edges slightly
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges

    def adjust_saturation(self, img, saturation_factor=1.3):
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        # Adjust saturation
        s = s * saturation_factor
        s = np.clip(s, 0, 255)
        
        # Ensure good brightness
        v = cv2.normalize(v, None, 50, 255, cv2.NORM_MINMAX)
        
        # Merge and convert back
        hsv = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return enhanced

    def apply_anime_effect(self, img):
        # Convert to float and normalize
        img_float = img.astype(np.float32) / 255.0
        
        # Edge preservation and smoothing
        bilateral = cv2.bilateralFilter(img_float, 5, 0.1, 1)
        
        # Create edge mask
        gray = cv2.cvtColor((bilateral * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edge = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 9, 2)
        
        # Thin edges
        kernel = np.ones((2,2), np.uint8)
        edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, kernel)
        
        # Convert edge back to RGB
        edge_rgb = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
        
        return bilateral, edge_rgb

    def cartoonify_image(self, style='classic'):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please open an image first")
            return

        try:
            # Make a copy
            img = self.original_image.copy()
            
            # Convert to RGB for better color processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply selected style
            style_functions = {
                'classic': apply_classic_cartoon,
                'comic': apply_comic_style,
                'watercolor': apply_watercolor_style,
                '3d': apply_3d_animation_style
            }
            
            if style not in style_functions:
                raise ValueError(f"Unknown style: {style}")
            
            result = style_functions[style](img_rgb)
            
            # Convert to BGR
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            
            # Update the cartoon image
            self.cartoon_image = result_bgr
            
            # Display the result
            self.display_image(self.cartoon_image, self.cartoon_display)

        except Exception as e:
            messagebox.showerror("Error", f"Error during cartoonification: {str(e)}")

    def setup_gui(self):
        # Create frames
        button_frame = tk.Frame(self.window)
        button_frame.pack(side=tk.TOP, pady=10)
        
        display_frame = tk.Frame(self.window)
        display_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Create buttons
        open_button = tk.Button(button_frame, text="Open Image", command=self.open_image)
        open_button.pack(side=tk.LEFT, padx=5)
        
        style_label = tk.Label(button_frame, text="Style:")
        style_label.pack(side=tk.LEFT, padx=5)
        
        style_var = tk.StringVar()
        style_var.set("classic")
        style_option = tk.OptionMenu(button_frame, style_var, "classic", "comic", "watercolor", "3d")
        style_option.pack(side=tk.LEFT, padx=5)
        
        cartoonify_button = tk.Button(button_frame, text="Cartoonify", command=lambda: self.cartoonify_image(style_var.get()))
        cartoonify_button.pack(side=tk.LEFT, padx=5)
        
        save_button = tk.Button(button_frame, text="Save Image", command=self.save_image)
        save_button.pack(side=tk.LEFT, padx=5)
        
        # Create image display areas
        self.original_display = tk.Label(display_frame)
        self.original_display.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        self.cartoon_display = tk.Label(display_frame)
        self.cartoon_display.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            try:
                # Read image
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise ValueError("Failed to load image")
                
                # Display original image
                self.display_image(self.original_image, self.original_display)
                
                # Reset cartoon image
                self.cartoon_image = None
                
            except Exception as e:
                messagebox.showerror("Error", f"Error opening image: {str(e)}")

    def save_image(self):
        if self.cartoon_image is None:
            messagebox.showwarning("Warning", "No cartoon image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            try:
                cv2.imwrite(file_path, self.cartoon_image)
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving image: {str(e)}")

    def display_image(self, image, display_label):
        if image is None:
            return
        
        # Get display size
        display_width = self.window.winfo_width() // 2 - 30
        display_height = self.window.winfo_height() - 100
        
        # Calculate scaling factor
        height, width = image.shape[:2]
        scale = min(display_width/width, display_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to RGB for display
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        pil_image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update label
        display_label.configure(image=photo)
        display_label.image = photo  # Keep a reference

    def run(self):
        self.window.mainloop()

def apply_classic_cartoon(img):
    # Convert to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Smoothing and edge preservation
    img_smooth = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Edge detection
    gray = cv2.cvtColor(img_smooth, cv2.COLOR_RGB2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY, 9, 9)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Color quantization
    n_colors = 8
    data = np.float32(img_smooth).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, 
                                  cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(img_smooth.shape)
    
    # Combine edges with colors
    result = cv2.bitwise_and(quantized, 255 - edges)
    
    return result

def apply_comic_style(img):
    # Initial processing
    img_smooth = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Enhanced edge detection
    gray = cv2.cvtColor(img_smooth, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Dramatic color quantization
    n_colors = 4  # Fewer colors for comic effect
    data = np.float32(img_smooth).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, 
                                  cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(img_smooth.shape)
    
    # Add comic-style effects
    result = cv2.addWeighted(quantized, 0.7, edges, 0.3, 0)
    
    # Enhance contrast
    result_pil = Image.fromarray(result)
    enhancer = ImageEnhance.Contrast(result_pil)
    result_pil = enhancer.enhance(1.5)
    
    return np.array(result_pil)

def apply_watercolor_style(img):
    # Soft blur for watercolor effect
    img_smooth = cv2.bilateralFilter(img, 9, 75, 75)
    img_smooth = cv2.GaussianBlur(img_smooth, (7, 7), 0)
    
    # Color abstraction
    n_colors = 6
    data = np.float32(img_smooth).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, 
                                  cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(img_smooth.shape)
    
    # Convert to PIL for watercolor effects
    result_pil = Image.fromarray(quantized)
    
    # Apply watercolor-like filters
    result_pil = result_pil.filter(ImageFilter.SMOOTH_MORE)
    result_pil = result_pil.filter(ImageFilter.EdgeEnhance_More)
    
    # Color enhancement
    enhancer = ImageEnhance.Color(result_pil)
    result_pil = enhancer.enhance(1.2)
    
    # Softness
    result_pil = result_pil.filter(ImageFilter.GaussianBlur(radius=1))
    
    return np.array(result_pil)

def apply_3d_animation_style(img):
    # Initial smoothing
    img_smooth = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Enhanced color segmentation
    n_colors = 12
    data = np.float32(img_smooth).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, 
                                  cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(img_smooth.shape)
    
    # Convert to PIL for 3D-like effects
    result_pil = Image.fromarray(quantized)
    
    # Enhance sharpness for 3D look
    enhancer = ImageEnhance.Sharpness(result_pil)
    result_pil = enhancer.enhance(1.5)
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(result_pil)
    result_pil = enhancer.enhance(1.2)
    
    # Color boost
    enhancer = ImageEnhance.Color(result_pil)
    result_pil = enhancer.enhance(1.3)
    
    # Final smoothing
    result_pil = result_pil.filter(ImageFilter.SMOOTH_MORE)
    
    return np.array(result_pil)

if __name__ == "__main__":
    app = ImageCartoonifier()
    app.run()
