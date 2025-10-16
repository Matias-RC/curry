import os
from xml.etree import ElementTree as ET
from level_generation import *
import numpy as np
# === Configuration ===
CELL_SIZE = 25  # Each cell is 100x100 pixels
TEMPLATE_MAP = {
    0: "floor.svg",
    1: "wall_inner.svg",
    2: "player.svg",
    3: "box.svg",
    4: "button.svg",
    5: "box_overbutton.svg",
    6: "player_on_button.svg",
    7: "wall.svg",
    8: "box_locked.svg"
}

def pretty_vis(matrix):
    matrix = np.array(matrix)
    rows, cols = matrix.shape
    result = matrix.copy()
    directions = [(-1,  0),( 1,  0),( 0, -1),( 0,  1), (-1, -1),(-1,  1), ( 1, -1),( 1,  1)] 
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if matrix[i, j] == 1:
                if all(matrix[i + dx, j + dy] == 1 for dx, dy in directions):
                    result[i, j] = 7

    return result



# === Helper: extract viewBox and compute scaling ===
def parse_viewbox(svg_root):
    """Return (min_x, min_y, width, height) from the SVG viewBox or defaults."""
    viewbox = svg_root.attrib.get("viewBox")
    if viewbox:
        parts = list(map(float, viewbox.strip().split()))
        if len(parts) == 4:
            return parts  # min_x, min_y, width, height
    # Fallback: guess from width/height if present
    width = float(svg_root.attrib.get("width", CELL_SIZE))
    height = float(svg_root.attrib.get("height", CELL_SIZE))
    return (0, 0, width, height)


# === Helper: wrap inner SVG contents as a centered <g> ===
def wrap_svg_as_group(svg_content, x_offset, y_offset):
    try:
        root = ET.fromstring(svg_content)
    except ET.ParseError as e:
        print("Error parsing SVG:", e)
        raise

    # Parse viewBox to know its original size
    min_x, min_y, vb_width, vb_height = parse_viewbox(root)

    # Scaling factor so tile fits inside CELL_SIZE
    scale = min(CELL_SIZE / vb_width, CELL_SIZE / vb_height)

    # Centering offsets (within one cell)
    offset_x = x_offset + (CELL_SIZE - vb_width * scale) / 2 - min_x * scale
    offset_y = y_offset + (CELL_SIZE - vb_height * scale) / 2 - min_y * scale

    # Create group
    group = ET.Element("g", {
        "transform": f"translate({offset_x},{offset_y}) scale({scale})"
    })

    # Append children of <svg> (not the <svg> element itself)
    for child in list(root):
        group.append(child)

    return group


# === Main: assemble final SVG ===
def generate_svg_grid(matrix, template_map, output_file="output.svg"):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    width = cols * CELL_SIZE
    height = rows * CELL_SIZE

    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": str(width),
        "height": str(height),
        "viewBox": f"0 0 {width} {height}",
    })

    # Cache SVG templates
    template_cache = {}
    for key, filename in template_map.items():
        with open(filename, "r", encoding="utf-8") as f:
            template_cache[key] = f.read()

    # Build grid
    for row_idx, row in enumerate(matrix):
        for col_idx, cell_value in enumerate(row):
            if cell_value not in template_cache:
                continue
            svg_content = template_cache[cell_value]
            group = wrap_svg_as_group(svg_content, col_idx * CELL_SIZE, row_idx * CELL_SIZE)
            svg.append(group)

    # Write to file
    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"SVG generated and saved to: {output_file}")


# === Run ===
if __name__ == "__main__":
    matrix = matrix_repr(generate_maze(8,8),8,8)
    generate_svg_grid(matrix, TEMPLATE_MAP, "empty_maze.svg")
    matrix = prim_longer_halls(25,25, 0.8)
    generate_svg_grid(matrix.tolist(), TEMPLATE_MAP, "empty_prim.svg")
    matrix_1 = easy_to_use_halls_without_connect(matrix.copy())
    generate_svg_grid(matrix_1.tolist(), TEMPLATE_MAP, "boxes_prim.svg")
    matrix_2 = easy_to_use_halls_connect(matrix.copy())
    matrix_2 = pretty_vis(matrix_2)
    generate_svg_grid(matrix_2.tolist(), TEMPLATE_MAP, "connected_prim.svg")
    CELL_SIZE = 50
    matrix = [
        [1,1,1,1],
        [1,8,0,1],
        [1,2,0,1],
        [1,4,0,1],
        [1,1,1,1]]
    generate_svg_grid(matrix, TEMPLATE_MAP, "wrong_box_pos.svg")
    matrix = [
        [1,1,1,1,1,1],
        [1,0,0,0,0,1],
        [1,0,0,0,0,1],
        [1,0,3,1,4,1],
        [1,0,0,0,0,1],
        [1,1,1,1,1,1]
    ]
    generate_svg_grid(matrix, TEMPLATE_MAP, "temp1.svg")
    matrix = [
        [1,1,1,1,1,1],
        [1,0,0,0,0,1],
        [1,0,0,1,0,1],
        [1,0,3,1,4,1],
        [1,0,0,0,0,1],
        [1,1,1,1,1,1]
    ]
    generate_svg_grid(matrix, TEMPLATE_MAP, "temp2.svg")
    matrix = [
        [1,1,1,1,1,1],
        [1,0,0,0,0,1],
        [1,0,0,1,0,1],
        [1,0,3,1,4,1],
        [1,0,0,1,0,1],
        [1,0,0,0,0,1],
        [1,1,1,1,1,1]
    ]
    generate_svg_grid(matrix, TEMPLATE_MAP, "temp3.svg")
    matrix = [
        [1,1,1,1,1,1],
        [1,0,0,0,0,1],
        [1,0,3,1,0,1],
        [1,0,0,4,0,1],
        [1,0,0,0,0,1],
        [1,1,1,1,1,1]
    ]
    generate_svg_grid(matrix, TEMPLATE_MAP, "temp4.svg")
    matrix = [
        [1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1],
        [1,0,3,1,1,0,0,1],
        [1,0,0,1,4,0,0,1],
        [1,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1]
    ]
    generate_svg_grid(matrix, TEMPLATE_MAP, "temp5.svg")
    CELL_SIZE = 25
    matrix = nxmfor_k_temps(20,20, templates, 7)
    generate_svg_grid(matrix, TEMPLATE_MAP, "nxm_k.svg")
    matrix = n_path(30)
    generate_svg_grid(matrix, TEMPLATE_MAP, "n_path.svg")
    matrix = simple_generate(20,20,7,20)
    generate_svg_grid(matrix, TEMPLATE_MAP, "randomized.svg")