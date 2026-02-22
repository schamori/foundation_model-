import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Define the path to your JSON file
JSON_FILE = Path("GraSP/TAPIS/outputs/custom_infer/epoch_0_preds_phases.json")
OUTPUT_CHART_FILE = Path("phase_predictions_chart.png")


def visualize_phases(json_path: Path, output_path: Path):
    """
    Loads phase predictions from JSON and generates a stacked area chart.
    """
    print(f"Loading predictions from {json_path}...")
    if not json_path.exists():
        print(f"Error: File not found at {json_path}", file=sys.stderr)
        return

    try:
        with json_path.open("r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}", file=sys.stderr)
        return
    except Exception as e:
        print(f"An error occurred loading the file: {e}", file=sys.stderr)
        return

    # Extract the 'phases_score_dist' list for each frame
    # This creates a new dictionary: {"frame_name": [dist_list]}
    processed_data = {
        frame: values["phases_score_dist"]
        for frame, values in data.items()
        if "phases_score_dist" in values
    }

    if not processed_data:
        print("Error: No 'phases_score_dist' data found in JSON.", file=sys.stderr)
        return

    # Convert the dictionary to a pandas DataFrame
    # 'orient="index"' makes frame names (the keys) the rows
    df = pd.DataFrame.from_dict(processed_data, orient="index")

    # Sort the DataFrame by its index (frame name)
    # This ensures "001.jpg" comes before "002.jpg"
    df = df.sort_index()

    # Rename columns for a clean legend
    df.columns = [f"Phase {i}" for i in range(df.shape[1])]

    # Reset the index to use a simple frame number (0, 1, 2...)
    # This makes the x-axis cleaner
    df = df.reset_index(drop=True)

    # --- Create the Plot ---
    print("Generating stacked area chart...")

    # 'df.plot.area' creates the chart
    # We get the 'ax' (axes) object to customize it
    ax = df.plot.area(
        stacked=True,
        colormap="viridis",  # 'viridis' is a common colormap
        linewidth=0,  # No lines between area segments
        figsize=(14, 7)  # Set figure size
    )

    # Set labels and title
    ax.set_title("Predicted Phase Distribution Over Time", fontsize=16)
    ax.set_xlabel("Frame Number", fontsize=12)
    ax.set_ylabel("Probability Score (Stacked)", fontsize=12)

    # Set Y-axis limit from 0 to 1 (since it's probability)
    ax.set_ylim(0, 1.0)

    # Set X-axis limit to match frame count
    ax.set_xlim(0, len(df) - 1)

    # Customize legend
    # Place legend outside the plot area
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="Phases")

    # Adjust layout to prevent the legend from being cut off
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the figure to a file
    try:
        plt.savefig(output_path)
        print(f"Success: Chart saved to {output_path}")
    except Exception as e:
        print(f"Error saving chart: {e}", file=sys.stderr)

    # Uncomment the next line if you want to display the plot
    # plt.show()


if __name__ == "__main__":
    visualize_phases(JSON_FILE, OUTPUT_CHART_FILE)