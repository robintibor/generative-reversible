import itertools
from PIL import Image
import numpy as np

def create_bw_image(image_cells):
    rows = image_cells.shape[0]
    cols = image_cells.shape[1]
    blank_image = Image.new('L', (image_cells.shape[3]*cols, image_cells.shape[2]*rows))
    for i_row, i_col in itertools.product(range(rows), range(cols)):
        x = image_cells[i_row, i_col]
        x = np.clip(255 - np.round(x * 255), 0, 255).astype(np.uint8)
        blank_image.paste(Image.fromarray(x), (i_col*image_cells.shape[3], i_row*image_cells.shape[2]))
    return blank_image


def create_rgb_image(image_cells):
    rows = image_cells.shape[0]
    cols = image_cells.shape[1]
    blank_image = Image.new('RGB', (image_cells.shape[4]*cols, image_cells.shape[3]*rows))
    for i_row, i_col in itertools.product(range(rows), range(cols)):
        x = image_cells[i_row, i_col]
        x = np.clip(np.round(x * 255), 0, 255).astype(np.uint8).transpose(1,2,0)
        blank_image.paste(Image.fromarray(x), (i_col*image_cells.shape[4], i_row*image_cells.shape[3]))
    return blank_image
