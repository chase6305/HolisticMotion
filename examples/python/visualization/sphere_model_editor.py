"""Interactive Viser editor for fitting link-local URDF collision spheres."""

import argparse
import time
from pathlib import Path

import numpy as np
import viser
from holistic_motion.geometry import (
    SphereFitOptions,
    fit_trimesh,
    load_sphere_model,
    load_urdf_collision_meshes,
    save_sphere_model,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, action="append", default=[])
    parser.add_argument("--port", type=int, default=8086)
    args = parser.parse_args()

    meshes = load_urdf_collision_meshes(
        args.urdf, package_dirs=args.package_dir
    )
    link_names = tuple(sorted(meshes))
    if args.output.is_file():
        model_links, metadata = load_sphere_model(args.output)
        model_links = dict(model_links)
    else:
        model_links, metadata = {}, {}
    fit_metrics = {}
    sphere_handles = []
    server = viser.ViserServer(port=args.port)
    selected_link = server.gui.add_dropdown(
        "Collision link", link_names, initial_value=link_names[0]
    )
    max_spheres = server.gui.add_number(
        "Maximum spheres", initial_value=24, min=1, max=300, step=1
    )
    min_radius = server.gui.add_number(
        "Minimum radius (mm)", initial_value=2.0, min=0.01, max=200.0, step=0.5
    )
    padding = server.gui.add_number(
        "Padding (mm)", initial_value=2.0, min=0.0, max=100.0, step=0.5
    )
    pitch = server.gui.add_number(
        "Voxel pitch (mm, 0=auto)", initial_value=0.0,
        min=0.0, max=100.0, step=0.5,
    )
    surface_samples = server.gui.add_number(
        "Surface samples", initial_value=5000, min=100, max=100000, step=100
    )
    coverage_mode = server.gui.add_checkbox(
        "Sampled coverage", initial_value=True
    )
    fit_button = server.gui.add_button("Fit selected link")
    fit_all_button = server.gui.add_button("Fit all links")
    save_button = server.gui.add_button("Save fitted links")
    status = server.gui.add_markdown("### Sphere Model Editor\nReady to fit.")
    mesh_handle = [None]

    def show_mesh(link):
        if mesh_handle[0] is not None:
            mesh_handle[0].remove()
        mesh_handle[0] = server.scene.add_mesh_trimesh(
            "/collision_mesh", meshes[link], opacity=0.3
        )

    def clear_spheres():
        for handle in sphere_handles:
            handle.remove()
        sphere_handles.clear()

    def show_spheres(link, spheres, metrics=None, mode="loaded"):
        clear_spheres()
        for index, sphere in enumerate(spheres):
            sphere_handles.append(
                server.scene.add_icosphere(
                    f"/fitted_spheres/{index:03d}",
                    radius=sphere.radius,
                    position=np.asarray(sphere.center),
                    color=(45, 170, 255) if index % 2 == 0 else (255, 155, 40),
                    opacity=0.45,
                )
            )
        if metrics is None:
            status.content = (
                f"### {link}\nMode: `{mode}`  \n"
                f"Spheres: **{len(spheres)}**  \n"
                "Refit this link to recompute quality metrics."
            )
        else:
            status.content = (
                f"### {link}\n"
                f"Mode: `{mode}`  \n"
                f"Spheres: **{metrics.sphere_count}**  \n"
                f"Sampled coverage: **{metrics.sampled_coverage:.2%}**  \n"
                f"Mean / maximum gap: "
                f"**{metrics.mean_uncovered_distance * 1000:.3f} / "
                f"{metrics.maximum_uncovered_distance * 1000:.3f} mm**"
            )

    def fit_selected():
        link = selected_link.value
        options = SphereFitOptions(
            max_spheres=int(max_spheres.value),
            min_radius=float(min_radius.value) / 1000.0,
            padding=float(padding.value) / 1000.0,
            sampled_coverage=bool(coverage_mode.value),
        )
        resolved_pitch = float(pitch.value) / 1000.0
        result = fit_trimesh(
            meshes[link],
            options,
            pitch=resolved_pitch if resolved_pitch > 0 else None,
            surface_samples=int(surface_samples.value),
        )
        model_links[link] = result.spheres
        fit_metrics[link] = (result.metrics, result.mode)
        show_spheres(link, result.spheres, result.metrics, result.mode)

    @selected_link.on_update
    def _change_link(event):
        show_mesh(event.target.value)
        if event.target.value in model_links:
            metric_entry = fit_metrics.get(event.target.value)
            show_spheres(
                event.target.value,
                model_links[event.target.value],
                metric_entry[0] if metric_entry else None,
                metric_entry[1] if metric_entry else "loaded",
            )
        else:
            clear_spheres()
            status.content = f"### {event.target.value}\nNot fitted yet."

    @fit_button.on_click
    def _fit(_event):
        try:
            status.content = f"### {selected_link.value}\nFitting..."
            fit_selected()
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            status.content = f"### Fit failed\n`{type(error).__name__}: {error}`"

    @fit_all_button.on_click
    def _fit_all(_event):
        completed = 0
        try:
            for link in link_names:
                selected_link.value = link
                status.content = (
                    f"### Fitting all links\n{completed}/{len(link_names)}: `{link}`"
                )
                fit_selected()
                completed += 1
            status.content += f"  \nCompleted **{completed}** links."
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            status.content = (
                f"### Fit all stopped\nCompleted {completed}/{len(link_names)}.  \n"
                f"`{type(error).__name__}: {error}`"
            )

    @save_button.on_click
    def _save(_event):
        if not model_links:
            status.content = "### Nothing to save\nFit at least one link first."
            return
        try:
            save_sphere_model(
                args.output,
                model_links,
                metadata={**metadata, "source_urdf": str(args.urdf.resolve())},
            )
            status.content += (
                f"  \nSaved **{len(model_links)}** links to `{args.output}`."
            )
        except (OSError, TypeError, ValueError) as error:
            status.content = (
                f"### Save failed\n`{type(error).__name__}: {error}`"
            )

    show_mesh(link_names[0])
    if link_names[0] in model_links:
        show_spheres(link_names[0], model_links[link_names[0]])
    print(f"Sphere Model Editor: http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
