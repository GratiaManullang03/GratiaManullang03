"""Render the gh-skyline STL model into a PNG that matches the profile theme.

Pure software rendering (numpy-stl + matplotlib), so it needs no OpenGL or
display and runs reliably on a headless CI runner.
"""

import numpy as np
from stl import mesh as stlmesh

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

STL_PATH = "skyline/contributions-skyline.stl"
PNG_PATH = "skyline/contributions-skyline.png"

BG = "#0f0524"
BASE_COLOR = np.array([0.545, 0.361, 0.965])  # violet, #8B5CF6


def main():
    model = stlmesh.Mesh.from_file(STL_PATH)
    triangles = model.vectors  # shape (N, 3, 3)

    # Shade each face by how much it faces a fixed light, so the city reads as 3D.
    normals = model.normals.astype(float)
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0
    normals = normals / lengths[:, None]

    light = np.array([0.4, -0.6, 0.75])
    light = light / np.linalg.norm(light)
    intensity = np.clip(normals.dot(light), 0.0, 1.0)
    intensity = 0.35 + 0.65 * intensity  # ambient floor so nothing goes black

    rgb = np.clip(intensity[:, None] * BASE_COLOR, 0.0, 1.0)
    facecolors = np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)

    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)

    collection = Poly3DCollection(
        triangles,
        facecolors=facecolors,
        edgecolors=(0, 0, 0, 0.12),
        linewidths=0.05,
    )
    ax.add_collection3d(collection)

    points = triangles.reshape(-1, 3)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins)

    ax.view_init(elev=28, azim=-55)
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(
        PNG_PATH,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.1,
    )
    print("rendered", PNG_PATH)


if __name__ == "__main__":
    main()
