"""
This example shows how to scale a model based on a generic model and a CSV static trial.
"""

import logging
import os
from pathlib import Path

import numpy as np

import biorbd
from biobuddy import (
    BiomechanicalModelReal,
    CsvData,
    ScaleTool,
    SegmentScaling,
    SegmentWiseScaling,
    Translations,
)


def main(visualization):

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # logging.FileHandler("app.log"),  # Log to a file
            logging.StreamHandler()  # Log to the console
        ],
    )

    # --- Paths --- #
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Static trial
    static_filepath = parent_path + "/examples/data/02_static.csv"
    csv_data = CsvData(
        csv_path=static_filepath,
    )

    # Paths
    current_path_file = Path(__file__).parent
    biomod_filepath = f"{current_path_file}/models/Arm_Projet.bioMod"
    scaled_biomod_filepath = f"{current_path_file}/models/Arm_scaled.bioMod"

    # Read an .osim file
    model = BiomechanicalModelReal().from_biomod(filepath=biomod_filepath)

    scale_tool = ScaleTool(original_model=model)
    scale_tool.add_scaling_segment(
        SegmentScaling(
            name="Arm",
            scaling_type=SegmentWiseScaling(
                axis=Translations.XYZ,
                marker_pairs=[
                    ["SA_3", "ELB_M"],
                ],
            ),
        )
    )
    scale_tool.add_scaling_segment(
        SegmentScaling(
            name="LowerArm1",
            scaling_type=SegmentWiseScaling(
                axis=Translations.XYZ,
                marker_pairs=[
                    ["WRB", "ELB_M"],
                ],
            ),
        )
    )
    scale_tool.add_scaling_segment(
        SegmentScaling(
            name="LowerArm2",
            scaling_type=SegmentWiseScaling(
                axis=Translations.XYZ,
                marker_pairs=[
                    ["WRA", "ELB_M"],
                ],
            ),
        )
    )

    # Scale the model
    scaled_model = scale_tool.scale(
        static_trial=csv_data,
        mass=70,
        q_regularization_weight=0.01,
        make_static_pose_the_models_zero=False,
        visualize_optimal_static_pose=False,
    )

    # Write the scaled model to a .bioMod file
    scaled_model.to_biomod(scaled_biomod_filepath, with_mesh=True)

    # Test that the model created is valid
    biorbd.Model(scaled_biomod_filepath)

    if visualization:
        import pyorerun

        # Compare the result visually
        t = np.linspace(0, 1, 10)
        viz = pyorerun.PhaseRerun(t)
        q = np.zeros((42, 10))

        # Biorbd model translated from .osim
        viz_biomod_model = pyorerun.BiorbdModel(biomod_filepath)
        viz_biomod_model.options.transparent_mesh = False
        viz_biomod_model.options.show_gravity = True
        viz_biomod_model.options.show_marker_labels = False
        viz_biomod_model.options.show_center_of_mass_labels = False
        viz.add_animated_model(viz_biomod_model, q)

        # Add the experimental markers from the static trial
        fake_exp_markers = np.repeat(scale_tool.mean_experimental_markers[:, :, np.newaxis], 10, axis=2)
        pyomarkers = pyorerun.PyoMarkers(data=fake_exp_markers, channels=scaled_model.marker_names, show_labels=False)

        # Model output
        viz_scaled_model = pyorerun.BiorbdModel(scaled_biomod_filepath)
        viz_scaled_model.options.transparent_mesh = False
        viz_scaled_model.options.show_gravity = True
        viz_scaled_model.options.show_marker_labels = False
        viz_scaled_model.options.show_center_of_mass_labels = False
        viz.add_animated_model(viz_scaled_model, q, tracked_markers=pyomarkers)

        # Animate
        viz.rerun_by_frame("Model output")


if __name__ == "__main__":
    try:
        import pyorerun

        visualization = True
    except:
        visualization = False

    main(visualization)
