#!/usr/bin/env python3
"""Generate a consolidated plot comparing train/test episode returns across agents.

Usage examples:
    # Plot all agents (default)
    python plot-results.py

    # Plot only specific agents
    python plot-results.py --agents "Neural - LSTM" "Neural - Transformer"

    # Plot symbolic KG variants only
    python plot-results.py --agents KG TKG

    # With states file and specific agents
    python plot-results.py --states path/to/states.yaml \\
        --agents "RDF-Star-NS (GCN)" "RDF-Star-NS (Transformer)"
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.lines import Line2D
import yaml
from collections import defaultdict

OUTPUT_DIR = Path("./figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_STATE_OUTPUT_DIR = Path("./figures/memory_state_graphs")
MEMORY_STATE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_CAPACITIES = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512]

AVAILABLE_AGENTS = [
    "Neural - LSTM",
    "Neural - Transformer",
    "KG",
    "TKG",
    "RDF-Star-NS (GCN)",
    "RDF-Star-NS (StarE-GCN)",
    "RDF-Star-NS (Transformer)",
]

DEFAULT_QA_PLOT_AGENTS = [
    "Neural - LSTM",
    "Neural - Transformer",
    "KG",
    "TKG",
]

FORGET_ACTION_LABELS = {0: "FIFO", 1: "LRU", 2: "LFU"}
QA_ACTION_LABELS = {0: "MRA", 1: "MRU", 2: "MFU"}
EXPLORE_ACTION_LABELS = QA_ACTION_LABELS

Q_VALUE_PLOTS = [
    ("forget", "Forget Q-Values", "q_values_forget.pdf", FORGET_ACTION_LABELS),
    ("qa", "QA Q-Values", "q_values_qa.pdf", QA_ACTION_LABELS),
    (
        "explore",
        "Explore Q-Values",
        "q_values_explore.pdf",
        EXPLORE_ACTION_LABELS,
    ),
]

DATA = {
    "train": {
        "Neural - LSTM": {
            0: {"mean": 8.0, "std": 0.0},
            2: {"mean": 11.6, "std": 2.417},
            4: {"mean": 10.8, "std": 1.833},
            8: {"mean": 13.4, "std": 1.855},
            16: {"mean": 11.6, "std": 1.02},
            32: {"mean": 12.6, "std": 2.577},
            64: {"mean": 9.8, "std": 0.98},
            128: {"mean": 10.2, "std": 1.166},
            256: {"mean": 11.0, "std": 0.632},
            512: {"mean": 15.2, "std": 0.98},
        },
        "Neural - Transformer": {
            0: {"mean": 8.0, "std": 0.0},
            2: {"mean": 14.4, "std": 1.855},
            4: {"mean": 15.2, "std": 0.748},
            8: {"mean": 15.4, "std": 1.02},
            16: {"mean": 15.6, "std": 1.356},
            32: {"mean": 16.8, "std": 0.748},
            64: {"mean": 16.0, "std": 1.265},
            128: {"mean": 16.8, "std": 0.748},
            256: {"mean": 16.0, "std": 2.449},
            512: {"mean": 17.6, "std": 3.072},
        },
        "KG": {
            0: {"mean": 1.0, "std": 0.0},
            2: {"mean": 2.88, "std": 0.204},
            4: {"mean": 2.28, "std": 0.733},
            8: {"mean": 4.24, "std": 0.916},
            16: {"mean": 5.68, "std": 0.412},
            32: {"mean": 10.16, "std": 0.512},
            64: {"mean": 18.08, "std": 0.688},
            128: {"mean": 28.92, "std": 1.318},
            256: {"mean": 41.4, "std": 1.403},
            512: {"mean": 42.96, "std": 0.697},
        },
        "TKG": {
            0: {"mean": 2.0, "std": 0.4},
            2: {"mean": 3.04, "std": 0.941},
            4: {"mean": 3.2, "std": 0.31},
            8: {"mean": 5.08, "std": 1.921},
            16: {"mean": 7.84, "std": 0.871},
            32: {"mean": 12.84, "std": 1.155},
            64: {"mean": 21.48, "std": 1.275},
            128: {"mean": 32.04, "std": 1.843},
            256: {"mean": 41.8, "std": 1.824},
            512: {"mean": 44.48, "std": 1.23},
        },
        "RDF-Star-NS (GCN)": {
            0: {"mean": 3.56, "std": 0.408},
            2: {"mean": 4.44, "std": 0.427},
            4: {"mean": 4.88, "std": 0.204},
            8: {"mean": 6.88, "std": 0.776},
            16: {"mean": 8.12, "std": 0.711},
            32: {"mean": 13.4, "std": 0.912},
            64: {"mean": 20.16, "std": 2.014},
            128: {"mean": 31.2, "std": 0.885},
            256: {"mean": 43.0, "std": 0.335},
            512: {"mean": 46.72, "std": 0.744},
        },
        "RDF-Star-NS (StarE-GCN)": {
            0: {"mean": 3.68, "std": 0.271},
            2: {"mean": 4.2, "std": 0.283},
            4: {"mean": 5.2, "std": 0.335},
            8: {"mean": 6.48, "std": 0.349},
            16: {"mean": 8.68, "std": 0.688},
            32: {"mean": 13.48, "std": 0.845},
            64: {"mean": 22.12, "std": 1.608},
            128: {"mean": 33.68, "std": 1.709},
            256: {"mean": 42.92, "std": 1.446},
            512: {"mean": 47.16, "std": 0.709},
        },
        "RDF-Star-NS (Transformer)": {
            0: {"mean": 3.28, "std": 0.349},
            2: {"mean": 4.52, "std": 0.744},
            4: {"mean": 5.24, "std": 0.983},
            8: {"mean": 6.84, "std": 0.941},
            16: {"mean": 8.28, "std": 0.546},
            32: {"mean": 13.16, "std": 1.203},
            64: {"mean": 23.2, "std": 1.815},
            128: {"mean": 33.64, "std": 1.768},
            256: {"mean": 44.28, "std": 0.882},
            512: {"mean": 47.0, "std": 0.607},
        },
    },
    "test": {
        "Neural - LSTM": {
            0: {"mean": 8.0, "std": 0.0},
            2: {"mean": 11.0, "std": 2.683},
            4: {"mean": 8.0, "std": 0.632},
            8: {"mean": 13.4, "std": 1.356},
            16: {"mean": 10.4, "std": 1.855},
            32: {"mean": 8.8, "std": 5.307},
            64: {"mean": 7.6, "std": 1.625},
            128: {"mean": 7.6, "std": 2.332},
            256: {"mean": 8.8, "std": 1.72},
            512: {"mean": 11.2, "std": 2.561},
        },
        "Neural - Transformer": {
            0: {"mean": 8.0, "std": 0.0},
            2: {"mean": 10.0, "std": 3.033},
            4: {"mean": 12.0, "std": 1.414},
            8: {"mean": 13.8, "std": 1.6},
            16: {"mean": 12.4, "std": 3.2},
            32: {"mean": 14.0, "std": 1.673},
            64: {"mean": 13.8, "std": 1.166},
            128: {"mean": 11.8, "std": 3.187},
            256: {"mean": 13.8, "std": 2.482},
            512: {"mean": 9.0, "std": 2.828},
        },
        "KG": {
            0: {"mean": 0.0, "std": 0.0},
            2: {"mean": 3.2, "std": 0.78},
            4: {"mean": 2.84, "std": 0.731},
            8: {"mean": 3.68, "std": 0.627},
            16: {"mean": 6.0, "std": 1.051},
            32: {"mean": 9.2, "std": 1.131},
            64: {"mean": 16.08, "std": 0.515},
            128: {"mean": 26.56, "std": 1.878},
            256: {"mean": 41.04, "std": 0.889},
            512: {"mean": 42.08, "std": 0.845},
        },
        "TKG": {
            0: {"mean": 2.0, "std": 0.669},
            2: {"mean": 2.32, "std": 0.515},
            4: {"mean": 2.64, "std": 0.557},
            8: {"mean": 3.92, "std": 0.56},
            16: {"mean": 7.28, "std": 1.849},
            32: {"mean": 15.04, "std": 1.141},
            64: {"mean": 20.92, "std": 1.97},
            128: {"mean": 32.84, "std": 2.236},
            256: {"mean": 43.08, "std": 1.401},
            512: {"mean": 45.64, "std": 1.209},
        },
        "RDF-Star-NS (GCN)": {
            0: {"mean": 2.16, "std": 0.557},
            2: {"mean": 1.96, "std": 0.463},
            4: {"mean": 2.52, "std": 0.574},
            8: {"mean": 3.32, "std": 0.325},
            16: {"mean": 5.56, "std": 1.52},
            32: {"mean": 11.32, "std": 1.473},
            64: {"mean": 16.32, "std": 2.178},
            128: {"mean": 28.0, "std": 3.657},
            256: {"mean": 42.88, "std": 3.115},
            512: {"mean": 45.36, "std": 1.311},
        },
        "RDF-Star-NS (StarE-GCN)": {
            0: {"mean": 2.28, "std": 0.483},
            2: {"mean": 2.48, "std": 0.16},
            4: {"mean": 3.6, "std": 0.681},
            8: {"mean": 3.56, "std": 1.286},
            16: {"mean": 5.48, "std": 1.478},
            32: {"mean": 10.0, "std": 1.951},
            64: {"mean": 17.32, "std": 1.742},
            128: {"mean": 30.24, "std": 1.632},
            256: {"mean": 43.16, "std": 2.639},
            512: {"mean": 45.6, "std": 1.459},
        },
        "RDF-Star-NS (Transformer)": {
            0: {"mean": 2.56, "std": 0.408},
            2: {"mean": 2.64, "std": 0.557},
            4: {"mean": 2.92, "std": 0.574},
            8: {"mean": 3.72, "std": 0.483},
            16: {"mean": 5.4, "std": 0.8},
            32: {"mean": 10.04, "std": 1.722},
            64: {"mean": 20.24, "std": 2.815},
            128: {"mean": 30.24, "std": 1.759},
            256: {"mean": 43.0, "std": 2.614},
            512: {"mean": 47.48, "std": 1.746},
        },
    },
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for auxiliary plots."""

    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--states",
        dest="states_path",
        type=Path,
        help="Path to states_q_values_actions_test.yaml file.",
    )
    parser.add_argument(
        "--agents",
        dest="agents",
        type=str,
        nargs="+",
        default=None,
        choices=AVAILABLE_AGENTS,
        help="Select specific agents to plot. Default: all agents.",
    )
    return parser.parse_args()


def _load_states_yaml(states_path: Path | None) -> object | None:
    """Load the states YAML file if provided."""

    if not states_path:
        return None

    if not states_path.exists():
        print(f"States file not found: {states_path}")
        return None

    try:
        with states_path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream) or []
    except yaml.YAMLError as exc:
        print(f"Failed to parse states YAML '{states_path}': {exc}")
        return None


def _extract_action_series(q_entries: list[list]) -> list[np.ndarray]:
    """Convert nested Q-value entries into per-action series."""

    if not q_entries:
        return []

    sanitized: list[list] = []
    for entry in q_entries:
        vector: list = []
        if isinstance(entry, list):
            if entry and isinstance(entry[0], list):
                vector = entry[0]
            else:
                vector = entry
        elif entry is not None:
            vector = [entry]
        sanitized.append(vector)

    num_actions = max((len(vec) for vec in sanitized), default=0)
    series_per_action: list[np.ndarray] = []

    for action_idx in range(num_actions):
        action_series: list[float] = []
        for vec in sanitized:
            value = vec[action_idx] if action_idx < len(vec) else None
            action_series.append(np.nan if value is None else value)
        series_per_action.append(np.asarray(action_series, dtype=float))

    return series_per_action


def _plot_q_value_category(
    category: str,
    title: str,
    q_entries: list,
    label_map: dict[int, str],
    output_path: Path,
) -> None:
    """Render and save a Q-value line plot for a given policy category."""

    series_per_action = _extract_action_series(q_entries)
    if not series_per_action:
        print(f"No Q-values available for '{category}'. Skipping plot.")
        return

    x_axis = np.arange(series_per_action[0].shape[0])

    fig, ax = plt.subplots(figsize=(12, 12), facecolor="white")
    ax.set_facecolor("white")

    for action_idx, values in enumerate(series_per_action):
        indices = np.where(~np.isnan(values))[0]
        if indices.size == 0:
            continue

        segments = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        label = label_map.get(action_idx, f"action_{action_idx}")

        for segment_idx, segment in enumerate(segments):
            if segment.size == 0:
                continue

            x_values = x_axis[segment]
            y_values = values[segment]
            ax.plot(
                x_values,
                y_values,
                label=label if segment_idx == 0 else "_nolegend_",
                linewidth=2.0,
            )

    if x_axis.size:
        xtick_min = 0
        xtick_max = 100
        ax.set_xticks(np.arange(xtick_min, xtick_max + 1, 10))
        buffer = 2
        ax.set_xlim(xtick_min - buffer, xtick_max + buffer)

    ax.set_title(title, fontsize=25, fontweight="bold")
    ax.set_xlabel("Step", fontsize=25)
    ax.set_ylabel("Q-value", fontsize=25)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best", fontsize=30)
    ax.tick_params(axis="both", labelsize=20)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path.resolve()}")


def _extract_q_values_from_states_data(data: object) -> dict[str, list]:
    """Normalize q-value storage regardless of YAML format."""

    if data is None:
        return {cat: [] for cat, *_ in Q_VALUE_PLOTS}

    categories = {cat: [] for cat, *_ in Q_VALUE_PLOTS}

    if isinstance(data, dict):
        test_section = data.get("test")
        if isinstance(test_section, dict):
            for category in categories:
                entries = test_section.get(category)
                if isinstance(entries, list):
                    categories[category] = entries
            return categories
        return categories

    if isinstance(data, list):
        for record in data:
            if not isinstance(record, dict):
                continue
            for category in categories:
                key = f"q_values_{category}"
                values = record.get(key)
                if values is None:
                    continue
                if isinstance(values, list):
                    categories[category].extend(values)
                else:
                    categories[category].append(values)
        return categories

    return categories


def _extract_state_records(data: object) -> list[dict]:
    """Extract individual state records from the loaded YAML structure."""

    if data is None:
        return []

    records: list[dict] = []

    if isinstance(data, list):
        for record in data:
            if isinstance(record, dict) and record.get("state"):
                records.append(record)
        return records

    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                for record in value:
                    if isinstance(record, dict) and record.get("state"):
                        records.append(record)
        return records

    return records


def _infer_rooms_from_records(records: list[dict]) -> set[str]:
    rooms: set[str] = set()
    for record in records:
        for entry in record.get("state", []):
            if isinstance(entry, list) and len(entry) >= 3:
                subj, rel, obj = entry[:3]
                if rel in {"north", "east", "south", "west"}:
                    if isinstance(subj, str):
                        rooms.add(subj)
                    if isinstance(obj, str):
                        rooms.add(obj)
                if rel == "at_location" and isinstance(obj, str):
                    rooms.add(obj)
    return rooms


def _infer_objects_from_records(records: list[dict], rooms: set[str]) -> set[str]:
    objects: set[str] = set()
    for record in records:
        for entry in record.get("state", []):
            if isinstance(entry, list) and len(entry) >= 3:
                subj, rel, _ = entry[:3]
                if rel == "at_location" and isinstance(subj, str) and subj != "agent":
                    objects.add(subj)
    # Ensure we don't classify rooms as objects
    return {obj for obj in objects if obj not in rooms}


NODE_COLOR_MAP = {
    "room": "#FFE4B5",  # Moccasin (Soft Gold)
    "static": "#87CEFA",  # Light Blue
    "moving": "#90EE90",  # Light Green
    "agent": "#D8BFD8",  # Thistle (Soft Purple)
    "wall": "#D3D3D3",  # Light Gray
    "unknown": "#FFFFFF",
}


def _load_environment_metadata(
    states_path: Path | None, records: list[dict]
) -> dict[str, set[str]]:
    """Load room/static/moving names from room-env configs with fallbacks."""

    metadata = {"rooms": set(), "static": set(), "moving": set()}

    inferred_rooms = _infer_rooms_from_records(records)
    inferred_objects = _infer_objects_from_records(records, inferred_rooms)

    def _load_config_metadata(config_path: Path) -> dict[str, set[str]] | None:
        try:
            with config_path.open("r", encoding="utf-8") as stream:
                config = json.load(stream)
            return {
                "rooms": set(config.get("room_names", [])),
                "static": set(config.get("static_names", [])),
                "moving": set(config.get("moving_names", [])),
            }
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Failed to load environment config '{config_path}': {exc}")
            return None

    def _score_candidate(candidate: dict[str, set[str]]) -> tuple[int, int]:
        candidate_objects = candidate.get("static", set()) | candidate.get("moving", set())
        object_overlap = len(inferred_objects & candidate_objects)
        room_overlap = len(inferred_rooms & candidate.get("rooms", set()))
        return (object_overlap, room_overlap)

    def _detect_environment_token(path: Path) -> str | None:
        segments = list(path.parts)
        segments.append(path.stem)
        for segment in segments:
            if "__" in segment:
                parts = segment.split("__")
                if parts:
                    candidate = parts[-1]
                    if candidate:
                        return candidate
        return None

    base_data_dir = (
        Path(__file__).resolve().parents[1] / "room-env" / "room_env" / "data"
    )
    candidate_paths: list[Path] = []

    if states_path is not None and base_data_dir.exists():
        token = _detect_environment_token(states_path)
        if token:
            pattern = f"room-config-{token}-v3.json"
            candidate = base_data_dir / pattern
            if candidate.exists():
                candidate_paths.append(candidate)

    if base_data_dir.exists():
        default_candidate = base_data_dir / "room-config-dev-v3.json"
        if default_candidate.exists():
            candidate_paths.append(default_candidate)

        for config_path in sorted(base_data_dir.glob("room-config-*-v3.json")):
            if config_path not in candidate_paths:
                candidate_paths.append(config_path)

    best_candidate: dict[str, set[str]] | None = None
    best_score = (-1, -1)
    for config_path in candidate_paths:
        loaded = _load_config_metadata(config_path)
        if loaded is None:
            continue
        score = _score_candidate(loaded)
        if score > best_score:
            best_score = score
            best_candidate = loaded

    if best_candidate is not None and best_score > (0, 0):
        metadata["rooms"].update(best_candidate.get("rooms", set()))
        metadata["static"].update(best_candidate.get("static", set()))
        metadata["moving"].update(best_candidate.get("moving", set()))

    metadata["rooms"].update(inferred_rooms)

    if metadata["static"] or metadata["moving"]:
        unresolved = inferred_objects - metadata["static"] - metadata["moving"]
        metadata["static"].update(unresolved)

    return metadata


def _categorize_node(node: str, metadata: dict[str, set[str]]) -> str:
    if node == "agent":
        return "agent"
    if node == "wall":
        return "wall"
    if node in metadata.get("rooms", set()):
        return "room"
    if node in metadata.get("moving", set()):
        return "moving"
    if node in metadata.get("static", set()):
        return "static"
    return "unknown"


def _aggregate_state_edges(state_entries: list) -> dict[tuple[str, str, str], int]:
    edge_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for entry in state_entries:
        if isinstance(entry, list) and len(entry) >= 3:
            subj, rel, obj = entry[:3]
            if isinstance(subj, str) and isinstance(rel, str) and isinstance(obj, str):
                edge_counts[(subj, rel, obj)] += 1
    return edge_counts


def _separate_overlapping_nodes(
    pos: dict, min_distance: float = 0.1, max_iterations: int = 50
) -> dict:
    """Separate overlapping networkx nodes by nudging their positions."""

    nodes = list(pos.keys())
    adjusted_pos = pos.copy()

    for _ in range(max_iterations):
        overlaps_found = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                pos1 = np.array(adjusted_pos[node1])
                pos2 = np.array(adjusted_pos[node2])

                distance = np.linalg.norm(pos1 - pos2)

                if distance < min_distance:
                    overlaps_found = True

                    if distance == 0:
                        angle = random.uniform(0, 2 * np.pi)
                        separation = (
                            np.array([np.cos(angle), np.sin(angle)]) * min_distance
                        )
                    else:
                        direction = (pos1 - pos2) / distance
                        separation = direction * (min_distance - distance) / 2

                    adjusted_pos[node1] = pos1 + separation
                    adjusted_pos[node2] = pos2 - separation

        if not overlaps_found:
            break

    return adjusted_pos


def _draw_memory_state_graph(
    edge_counts: dict[tuple[str, str, str], int],
    metadata: dict[str, set[str]],
    index: int,
    output_path: Path,
) -> None:
    if not edge_counts:
        return

    G = nx.DiGraph()
    for (subj, rel, obj), count in edge_counts.items():
        label = f"{rel} ({count})" if count > 1 else rel
        G.add_edge(subj, obj, label=label, weight=count)

    if G.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 12), facecolor="white")
    ax.set_facecolor("white")

    pos = nx.kamada_kawai_layout(G, scale=1.6)
    pos = _separate_overlapping_nodes(pos, min_distance=0.25, max_iterations=150)

    node_colors = []
    for node in G.nodes():
        category = _categorize_node(node, metadata)
        color = NODE_COLOR_MAP.get(category, NODE_COLOR_MAP["unknown"])
        node_colors.append(color)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=2600,
        alpha=0.9,
        edgecolors="black",
        linewidths=2,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=12,
        font_weight="bold",
        font_color="black",
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="gray",
        arrows=True,
        arrowsize=25,
        arrowstyle="->",
        alpha=0.7,
        width=2,
        connectionstyle="arc3,rad=0.1",
    )

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels,
        ax=ax,
        font_size=12,
        font_color="black",
        font_weight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
    )

    ax.set_title(
        f"Memory State {index:03d}",
        fontsize=16,
        fontweight="bold",
    )
    ax.axis("off")

    if pos:
        xs = np.fromiter((coord[0] for coord in pos.values()), dtype=float)
        ys = np.fromiter((coord[1] for coord in pos.values()), dtype=float)
        if xs.size and ys.size:
            x_span = xs.max() - xs.min()
            y_span = ys.max() - ys.min()
            pad_x = max(x_span * 0.15, 0.2)
            pad_y = max(y_span * 0.15, 0.2)
            ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
            ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)

    # fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.94)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path.resolve()}")


def _compute_coverage_metrics(records: list[dict]) -> tuple[list[int], list[int]]:
    """Compute room coverage and effective triple coverage for each timestep.
    
    Returns:
        - room_coverage: cumulative count of unique rooms visited (monotonic)
        - triple_coverage: count of unique triples in memory at each step
    """
    room_coverage = []
    triple_coverage = []
    visited_rooms = set()
    
    for record in records:
        state_entries = record.get("state", [])
        
        # Room coverage: cumulative unique rooms where (agent, at_location, room)
        for entry in state_entries:
            if isinstance(entry, list) and len(entry) >= 3:
                subj, rel, obj = entry[:3]
                if subj == "agent" and rel == "at_location" and isinstance(obj, str):
                    visited_rooms.add(obj)
        
        room_coverage.append(len(visited_rooms))
        
        # Triple coverage: unique (s, p, o) triples in current state
        unique_triples = set()
        for entry in state_entries:
            if isinstance(entry, list) and len(entry) >= 3:
                triple = (entry[0], entry[1], entry[2])
                unique_triples.add(triple)
        
        triple_coverage.append(len(unique_triples))
    
    return room_coverage, triple_coverage


def plot_coverage_metrics(states_data: object, states_path: Path | None) -> None:
    """Plot room coverage and effective triple coverage on dual y-axes."""
    
    records = _extract_state_records(states_data)
    if not records:
        print("No state records found; skipping coverage metrics plot.")
        return
    
    room_coverage, triple_coverage = _compute_coverage_metrics(records)
    
    if not room_coverage or not triple_coverage:
        print("No coverage data computed; skipping coverage metrics plot.")
        return
    
    steps = np.arange(len(room_coverage))
    
    # Determine agent type for title and filename
    agent_type = "Agent"
    if states_path:
        state_name = states_path.name.lower()
        if "rdf-star" in state_name or "symbolic_test" in state_name:
            agent_type = "TKG"
        elif "rdf" in state_name or "symbolic_simple" in state_name:
            agent_type = "KG"
        elif "lstm" in states_path.name.lower():
            agent_type = "Neural - LSTM"
        elif "transformer" in states_path.name.lower():
            agent_type = "Neural - Transformer"
        else:
            agent_type = "Unknown"

    # Save raw values to markdown
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)
    agent_slug = (
        agent_type.lower().replace(" - ", "_").replace("-", "_").replace(" ", "_")
    )
    md_filename = f"coverage_metrics_{agent_slug}.md"
    md_path = data_dir / md_filename
    
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Coverage Metrics - {agent_type} Agent\n\n")
        f.write("## Raw Values\n\n")
        f.write("| Step | Room Coverage | Triple Coverage |\n")
        f.write("|------|---------------|-----------------|\n")
        for step, room_cov, mem_cov in zip(steps, room_coverage, triple_coverage):
            f.write(f"| {step} | {room_cov} | {mem_cov} |\n")
        f.write("\n## Summary\n\n")
        f.write(f"- **Final Room Coverage**: {room_coverage[-1]}\n")
        f.write(f"- **Final Triple Coverage**: {triple_coverage[-1]}\n")
        f.write(f"- **Total Steps**: {len(steps)}\n")
    
    print(f"Coverage metrics data saved to {md_path.resolve()}")
    
    # Create plot with matching styling to agent_train_test_returns.pdf
    fig, ax1 = plt.subplots(figsize=(12, 8), facecolor="white")
    ax1.set_facecolor("white")
    
    # Room coverage on left y-axis (purple)
    color_room = "#9467bd"  # Purple
    ax1.set_xlabel("Step", fontsize=25)
    ax1.set_ylabel("Room Coverage", fontsize=25, color=color_room)
    line1 = ax1.plot(steps, room_coverage, color=color_room, linewidth=2.5, 
                     marker="o", markersize=4, label="Room Coverage")
    ax1.tick_params(axis="y", labelcolor=color_room, labelsize=20)
    ax1.tick_params(axis="x", labelsize=20)
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax1.set_ylim(0, 50)
    
    # Triple coverage on right y-axis (green)
    ax2 = ax1.twinx()
    color_triple = "#2ca02c"  # Green
    ax2.set_ylabel("Effective Triple Coverage", fontsize=25, color=color_triple)
    line2 = ax2.plot(steps, triple_coverage, color=color_triple, linewidth=2.5,
                     marker="s", markersize=4, linestyle="--", label="Triple Coverage")
    ax2.tick_params(axis="y", labelcolor=color_triple, labelsize=20)
    ax2.set_ylim(0, 512)
    
    # Title with bold font matching agent_train_test_returns.pdf
    ax1.set_title(f"Coverage Metrics Over Time - {agent_type} Agent", 
                  fontsize=25, fontweight="bold")
    
    # Combined legend
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=20, frameon=True, framealpha=0.9)
    
    fig.tight_layout()
    
    output_path = OUTPUT_DIR / f"coverage_metrics_{agent_slug}.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Coverage metrics plot saved to {output_path.resolve()}")
    print(f"  Final room coverage: {room_coverage[-1]}")
    print(f"  Final effective triple coverage: {triple_coverage[-1]}")


def render_memory_state_graphs(states_data: object, states_path: Path | None) -> None:
    """Render and save memory state graphs for each timestep."""

    records = _extract_state_records(states_data)
    if not records:
        print("No state records found; skipping memory state graph generation.")
        return

    metadata = _load_environment_metadata(states_path, records)

    unknown_nodes: set[str] = set()

    for idx, record in enumerate(records):
        if idx >= 100:
            break

        state_entries = record.get("state")
        if not isinstance(state_entries, list):
            continue

        edge_counts = _aggregate_state_edges(state_entries)
        if not edge_counts:
            continue

        for subj, _, obj in edge_counts:
            for node in (subj, obj):
                if _categorize_node(node, metadata) == "unknown":
                    unknown_nodes.add(node)

        output_path = MEMORY_STATE_OUTPUT_DIR / f"memory_state_{idx:03d}.pdf"
        _draw_memory_state_graph(edge_counts, metadata, idx, output_path)

    if unknown_nodes:
        formatted = ", ".join(sorted(unknown_nodes))
        print(
            "Warning: some nodes were not categorized and were rendered as 'unknown':"
            f" {formatted}"
        )


def plot_q_values_from_states(states_data: object, states_path: Path | None) -> None:
    """Produce per-policy Q-value plots using loaded states data."""

    if states_data is None:
        print("No states data available; skipping Q-value plots.")
        return

    q_values_by_category = _extract_q_values_from_states_data(states_data)

    for category, title, filename, label_map in Q_VALUE_PLOTS:
        q_entries = q_values_by_category.get(category)
        if not q_entries:
            source = states_path if states_path else "provided data"
            print(f"No data for '{category}' in {source}. Skipping plot.")
            continue

        output_path = OUTPUT_DIR / filename
        _plot_q_value_category(category, title, q_entries, label_map, output_path)


def extract_series(environment: str, agent: str) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and std arrays for the requested environment/agent."""
    env_data = DATA[environment][agent]
    means = [env_data[capacity]["mean"] for capacity in MEMORY_CAPACITIES]
    stds = [env_data[capacity]["std"] for capacity in MEMORY_CAPACITIES]
    return np.asarray(means, dtype=float), np.asarray(stds, dtype=float)


def main() -> None:
    args = parse_args()
    plt.style.use("default")

    states_path = args.states_path
    states_data = _load_states_yaml(states_path)

    if states_data is not None:
        plot_q_values_from_states(states_data, states_path)
        plot_coverage_metrics(states_data, states_path)
        render_memory_state_graphs(states_data, states_path)
    else:
        if states_path:
            print("Failed to load states data; skipping Q-value plots and graphs.")
        else:
            print("No states file provided; skipping Q-value plots and graphs.")

    # Default to the subset used in this paper; RDF-Star-NS variants belong to another paper.
    selected_agents = args.agents if args.agents else DEFAULT_QA_PLOT_AGENTS
    
    fig, ax = plt.subplots(figsize=(12, 12), facecolor="white")
    ax.set_facecolor("white")
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(selected_agents)))

    positions = np.arange(len(MEMORY_CAPACITIES), dtype=float)

    for idx, agent in enumerate(selected_agents):
        train_mean, train_std = extract_series("train", agent)
        test_mean, test_std = extract_series("test", agent)
        color = colors[idx]

        ax.plot(
            positions,
            train_mean,
            label=f"{agent} (train)",
            color=color,
            linewidth=2.5,
            marker="o",
        )
        ax.fill_between(
            positions,
            train_mean - train_std,
            train_mean + train_std,
            color=color,
            alpha=0.1,
        )

        ax.plot(
            positions,
            test_mean,
            label=f"{agent} (test)",
            color=color,
            linewidth=2.5,
            linestyle="--",
            marker="s",
        )
        ax.fill_between(
            positions,
            test_mean - test_std,
            test_mean + test_std,
            color=color,
            alpha=0.1,
        )

    ax.set_title(
        "Train vs Test QA Accuracy by Long-Term Memory Capacity",
        fontsize=25,
        fontweight="bold",
    )
    ax.set_xlabel("Long-Term Memory Capacity", fontsize=25)
    ax.set_ylabel("QA Accuracy", fontsize=25)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(capacity) for capacity in MEMORY_CAPACITIES])
    ax.tick_params(axis="both", labelsize=20)

    handles = []
    for idx, agent in enumerate(selected_agents):
        color = colors[idx]
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=2.5,
                    marker="o",
                    label=f"{agent} (train)",
                ),
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=2.5,
                    linestyle="--",
                    marker="s",
                    label=f"{agent} (test)",
                ),
            ]
        )

    ax.legend(
        handles=handles,
        ncol=1,
        fontsize=20,
        frameon=True,
        framealpha=0.9,
        loc="best",
    )
    fig.tight_layout()

    pdf_path = OUTPUT_DIR / "agent_train_test_qa_accuracy.pdf"

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Combined plot saved to {pdf_path}")


if __name__ == "__main__":
    main()
