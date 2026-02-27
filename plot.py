import os
import glob
from PIL import Image
import math


def aggregate_plots():
    """Aggregate multiple PNG plots into a single grid image."""

    # Find all matching PNG files
    search_pattern = os.path.join(RESULTS_DIR, FILE_PATTERN)
    png_files = glob.glob(search_pattern)

    if not png_files:
        print(f"No PNG files found matching pattern: {search_pattern}")
        return

    if SORT_FILES:
        png_files.sort()

    print(f"Found {len(png_files)} PNG files:")
    for f in png_files:
        print(f"  - {os.path.basename(f)}")

    # Load all images
    images = []
    for png_file in png_files:
        try:
            img = Image.open(png_file)
            images.append(img)
        except Exception as e:
            print(f"Error loading {png_file}: {e}")

    if not images:
        print("No images could be loaded.")
        return

    # Determine grid dimensions
    n_images = len(images)
    cols = GRID_COLS
    rows = GRID_ROWS if GRID_ROWS is not None else math.ceil(n_images / cols)

    print(f"\nCreating grid: {rows} rows × {cols} columns")

    # Get the maximum dimensions across all images
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Calculate total canvas size
    canvas_width = MARGIN * 2 + cols * max_width + (cols - 1) * PADDING
    canvas_height = MARGIN * 2 + rows * max_height + (rows - 1) * PADDING

    # Create blank canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), BACKGROUND_COLOR)

    # Paste images onto canvas
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        # Calculate position (center each image in its cell)
        x = MARGIN + col * (max_width + PADDING) + (max_width - img.width) // 2
        y = MARGIN + row * (max_height + PADDING) + (max_height - img.height) // 2

        canvas.paste(img, (x, y))

    # Save the aggregated image
    canvas.save(OUTPUT_FILE)
    print(f"\nAggregated plot saved to: {OUTPUT_FILE}")
    print(f"Final dimensions: {canvas_width} × {canvas_height} pixels")


if __name__ == "__main__":
    # ============== CONFIGURATION VARIABLES ==============
    # Directory containing the PNG files
    DATA = "m5"
    # DATA = "iot"

    RESULTS_DIR = f"./results_{DATA}/"

    # Pattern to match PNG files (e.g., '<model>_plot_tsne.png')
    FILE_PATTERN = "*_plot_tsne.png"

    # Grid configuration
    GRID_COLS = 3  # Number of columns in the grid
    GRID_ROWS = None  # Number of rows (None = auto-calculate based on number of images)

    # Spacing and margins
    PADDING = 20  # Pixels between images
    MARGIN = 40  # Pixels around the entire grid

    # Background color (RGB tuple)
    BACKGROUND_COLOR = (255, 255, 255)  # White

    # Output file
    OUTPUT_FILE = f"./.info/{DATA}_plots_tsne.png"

    # Sort files alphabetically (True/False)
    SORT_FILES = False

    aggregate_plots()