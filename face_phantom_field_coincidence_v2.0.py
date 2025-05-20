import os
import cv2
import numpy as np
import pydicom
import tkinter as tk
from tkinter import filedialog
from pylinac import FieldProfileAnalysis, Centering, Normalization, Edge
from pylinac.metrics.profile import (
    PenumbraLeftMetric,
    PenumbraRightMetric,
    SymmetryAreaMetric,
    FlatnessDifferenceMetric,
    CAXToLeftEdgeMetric,
    CAXToRightEdgeMetric,
)
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table
import csv

csv_results = []


def enhance_image(img_array):
    img = img_array.astype('uint16')
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_norm)
    img_blur = cv2.GaussianBlur(img_clahe, (5, 5), 0)
    return img_blur


def find_bb_markers(enhanced_img):
    circles = cv2.HoughCircles(
        enhanced_img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=20,
        param2=40,
        minRadius=1,
        maxRadius=50
    )
    if circles is not None:
        return np.uint16(np.around(circles[0]))
    return []


def extract_metadata(ds):
    return {
        'RT Image Description': ds.get('RTImageDescription', ''),
        'Radiation Machine Name': ds.get('RadiationMachineName', ''),
        'SAD': ds.get('RadiationMachineSAD', 'N/A'),
        'SID': ds.get('RTImageSID', 'N/A')
    }


def draw_results_table(ax, table_data, col_labels):
    ax.axis('off')
    table = Table(ax, bbox=[0, 0, 1, 1])
    cell_height = 1 / (len(table_data) + 1)
    cell_width = 1 / len(col_labels)

    for i, label in enumerate(col_labels):
        table.add_cell(0, i, width=cell_width, height=cell_height, text=label, loc='center', facecolor='lightgray')

    for row_idx, row in enumerate(table_data, start=1):
        for col_idx, val in enumerate(row):
            table.add_cell(row_idx, col_idx, width=cell_width, height=cell_height, text=str(val), loc='center')

    ax.add_table(table)


def process_and_plot(filepath, pdf):
    ds = pydicom.dcmread(filepath)
    img = ds.pixel_array.astype(float)
    px_spacing = ds.get('PixelSpacing', ds.get('ImagePlanePixelSpacing', [1.0, 1.0]))
    enhanced = enhance_image(img)
    metadata = extract_metadata(ds)
    sid = float(metadata.get('SID', 1500))
    sod = float(metadata.get('SAD', 1000))
    scale = sod / sid

    analyzer = FieldProfileAnalysis(filepath)
    analyzer.analyze(
        centering=Centering.BEAM_CENTER,
        x_width=0.02,
        y_width=0.02,
        normalization=Normalization.BEAM_CENTER,
        edge_type=Edge.INFLECTION_HILL,
        hill_window_ratio=0.05,
        ground=True,
        metrics=[
            PenumbraLeftMetric(),
            PenumbraRightMetric(),
            SymmetryAreaMetric(),
            FlatnessDifferenceMetric(),
            CAXToLeftEdgeMetric(),
            CAXToRightEdgeMetric(),
        ],
    )

    results = analyzer.results_data(as_dict=True)
    x_metrics = results.get('x_metrics', {})
    y_metrics = results.get('y_metrics', {})

    rad_cx_px = analyzer.x_profile.center_idx
    rad_cy_px = analyzer.y_profile.center_idx
    rad_cx_mm = rad_cx_px * px_spacing[1]
    rad_cy_mm = rad_cy_px * px_spacing[0]

    fig, axs = plt.subplots(2, 1, figsize=(6.5, 9.5), gridspec_kw={'height_ratios': [2.5, 2.5], 'hspace': 0.7})
    ax_img, ax_table = axs

    ax_img.imshow(img, cmap='gray')
    ax_img.axhline(rad_cy_px, color='red', linestyle='-', linewidth=1)
    ax_img.axvline(rad_cx_px, color='red', linestyle='-', linewidth=1)
    ax_img.plot(rad_cx_px, rad_cy_px, marker='+', color='red', markersize=8, markeredgewidth=1.5, label='Radiation Center')

    csv_row = {'Filename': os.path.basename(filepath)}
    max_diff = 0
    table_data = []
    col_labels = ['Edge', 'LF→E (mm)', 'Rad→E (mm)', 'Δ (mm)', 'Result']

    markers = find_bb_markers(enhanced)
    if markers is None or len(markers) < 4:
        print("Insufficient BBs")
        return

    top = sorted(markers, key=lambda m: m[1])[:2]
    bottom = sorted(markers, key=lambda m: m[1])[-2:]
    left = sorted(markers, key=lambda m: m[0])[:2]
    right = sorted(markers, key=lambda m: m[0])[-2:]

    def shifted_midpoint(pair, shift_mm, axis):
        x = np.mean([p[0] for p in pair])
        y = np.mean([p[1] for p in pair])
        shift_px = shift_mm / px_spacing[0 if axis == 'x' else 1] / scale
        if axis == 'x':
            x += shift_px
        else:
            y += shift_px
        return x, y

    top_x, top_y = shifted_midpoint(top, -15, 'y')
    bottom_x, bottom_y = shifted_midpoint(bottom, 15, 'y')
    left_x, left_y = shifted_midpoint(left, -15, 'x')
    right_x, right_y = shifted_midpoint(right, 15, 'x')

    lf_corners = {
        'Top Edge': ((left_x + right_x) / 2, top_y),
        'Bottom Edge': ((left_x + right_x) / 2, bottom_y),
        'Left Edge': (left_x, (top_y + bottom_y) / 2),
        'Right Edge': (right_x, (top_y + bottom_y) / 2)
    }

    lf_cx_px = (left_x + right_x) / 2
    lf_cy_px = (top_y + bottom_y) / 2
    lf_cx_mm = lf_cx_px * px_spacing[1]
    lf_cy_mm = lf_cy_px * px_spacing[0]

    ax_img.axhline(lf_cy_px, color='blue', linestyle='--', linewidth=1)
    ax_img.axvline(lf_cx_px, color='blue', linestyle='--', linewidth=1)
    ax_img.plot(lf_cx_px, lf_cy_px, marker='+', color='blue', markersize=8, markeredgewidth=1.5, label='Light Field Center')

    lf_box_coords = [
        (left_x, top_y), (right_x, top_y),
        (right_x, bottom_y), (left_x, bottom_y),
        (left_x, top_y)
    ]
    ax_img.plot(*zip(*lf_box_coords), linestyle=':', color='blue', linewidth=1.2, label='Light Field Box')

    for (x, y, r) in markers:
        ax_img.add_patch(plt.Circle((x, y), r, color='lime', fill=False, linestyle='--', linewidth=0.5))

    for label, (x_px, y_px) in lf_corners.items():
        x_mm = x_px * px_spacing[1]
        y_mm = y_px * px_spacing[0]
        dist_lf = np.hypot(x_mm - lf_cx_mm, y_mm - lf_cy_mm)
        dist_rad = np.hypot(x_mm - rad_cx_mm, y_mm - rad_cy_mm)
        delta = abs(dist_lf - dist_rad)
        result = 'PASS' if delta <= 2.0 else 'FAIL'
        max_diff = max(max_diff, delta)

        ax_img.plot(x_px, y_px, marker='+', color='#e1ad01', markersize=8, markeredgewidth=1.2)
        table_data.append([label, f"{dist_lf:.2f}", f"{dist_rad:.2f}", f"{delta:.2f}", result])

    ax_table.axis('off')
    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='semibold')
        else:
            cell.set_text_props(weight='normal')
            if col == 4:
                val = cell.get_text().get_text()
                if val == 'PASS':
                    cell.set_facecolor('#d0f0c0')  # light green
                elif val == 'FAIL':
                    cell.set_facecolor('#f8d7da')  # light red

    ax_img.axis('off')
    ax_img.legend(loc='lower right', fontsize=7)

    fig.subplots_adjust(left=0.1, right=0.9, top=0.65, bottom=0.06)
    fig.suptitle(
        f"Field Coincidence QA: {os.path.basename(filepath)}\n"
        f"Machine: {metadata['Radiation Machine Name']} | Energy: {metadata['RT Image Description']}",
        fontsize=9, y=0.80
    )

    import matplotlib.image as mpimg
    logo_path = 'SBM_logo.jpg'
    if os.path.exists(logo_path):
        logo = mpimg.imread(logo_path)
        fig.figimage(logo, xo=40, yo=int(fig.bbox.ymax) - 60, alpha=1, zorder=1)

    fig.text(
        0.82, 0.02,
        'Key: LF→E = Light Field center to edge, Rad→E = Radiation center to edge, Δ = Absolute difference',
        fontsize=6, ha='right'
    )
    pdf.savefig(fig)
    plt.close(fig)

    csv_row['QA Result'] = 'PASS' if max_diff <= 2.0 else 'FAIL'
    csv_row['Max Delta (mm)'] = f"{max_diff:.2f}"
    csv_results.append(csv_row)


def main():
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(filetypes=[("DICOM files", "*.dcm")], title="Select DICOM QA Images")
    if not files:
        print("No files selected.")
        return

    with PdfPages("FieldCoincidenceQA_Report.pdf") as pdf:
        for file in files:
            process_and_plot(file, pdf)

    if csv_results:
        keys = csv_results[0].keys()
        with open("FieldCoincidenceQA_Results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(csv_results)

    print("Analysis complete. Results saved to FieldCoincidenceQA_Report.pdf and FieldCoincidenceQA_Results.csv")


if __name__ == "__main__":
    main()