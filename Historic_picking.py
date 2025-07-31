import pandas as pd
import plotly.express as px
import os

# ─────────────────────────────────────────────────────────────
# Load and preprocess
# ─────────────────────────────────────────────────────────────
file_path = r"C:\Users\guillermolambertus.v\Thesis_project_code\Data\Training\Features_all_dates.csv"
df = pd.read_csv(file_path)

# Convert timestamps
df["START_PICKING_TS"] = pd.to_datetime(df["START_PICKING_TS"], errors='coerce')
df["END_PICKING_TS"] = pd.to_datetime(df["END_PICKING_TS"], errors='coerce')
df["PICK_DATE"] = pd.to_datetime(df["PICK_DATE"], errors='coerce')

# ─────────────────────────────────────────────────────────────
# Filter: Jan 24, 2025 and TEMPZONE = AMBIENT
# ─────────────────────────────────────────────────────────────
df = df[
    (df["PICK_DATE"] == "2025-01-24") &
    (df["TEMPZONE"].str.upper() == "AMBIENT")
].copy()

# ─────────────────────────────────────────────────────────────
# Group by PLANNED_FRAME_STACK_ID to find reliable start times
# ─────────────────────────────────────────────────────────────
fs_df = df.groupby("PLANNED_FRAME_STACK_ID").agg(
    ACTUAL_START_TS=("START_PICKING_TS", "min"),
    ACTUAL_END_TS=("END_PICKING_TS", "max"),
    SHIPMENT_ID=("TRUCK_TRIP_ID", "first"),
    PLANNED_PICKING_DEADLINE=("PLANNED_PICKING_DEADLINE_MINS", "first")
).reset_index()

# Drop rows with missing shipment or start
fs_df = fs_df.dropna(subset=["SHIPMENT_ID", "ACTUAL_START_TS"])

# ─────────────────────────────────────────────────────────────
# Select 11 shipments with earliest actual START time
# ─────────────────────────────────────────────────────────────
shipment_start_times = fs_df.groupby("SHIPMENT_ID")["ACTUAL_START_TS"].min().sort_values()
selected_early_shipments = shipment_start_times.head(11).index.tolist()

# Find the latest deadline among the selected ones
selected_deadlines = fs_df[fs_df["SHIPMENT_ID"].isin(selected_early_shipments)]
latest_deadline = selected_deadlines["PLANNED_PICKING_DEADLINE"].max()

# Find an additional shipment with a later deadline
remaining_shipments = fs_df[~fs_df["SHIPMENT_ID"].isin(selected_early_shipments)]
later_shipment_candidates = remaining_shipments.groupby("SHIPMENT_ID")["PLANNED_PICKING_DEADLINE"].first()
later_shipment_candidates = later_shipment_candidates[later_shipment_candidates > latest_deadline]

# Select one such shipment if available
if not later_shipment_candidates.empty:
    extra_shipment = later_shipment_candidates.sort_values().index[0]
    print(f"📦 Including one extra shipment with later deadline: {extra_shipment}")
    selected_shipments = selected_early_shipments + [extra_shipment]
else:
    print("⚠️ No shipment found with a later deadline. Using only the initial 11.")
    selected_shipments = selected_early_shipments

# Filter only frame-stacks belonging to selected shipments
agg_df = fs_df[fs_df["SHIPMENT_ID"].isin(selected_shipments)].copy()
agg_df = agg_df.sort_values("ACTUAL_START_TS")


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────
def visualize_frame_stack_schedule(agg_df, output_path="Historic_output/historic_frame_stack_schedule.html"):
    base_day = pd.to_datetime("2025-01-24")

    # Parse deadlines
    agg_df["PLANNED_PICKING_DEADLINE_TS"] = agg_df["PLANNED_PICKING_DEADLINE"].apply(
        lambda mins: base_day + pd.Timedelta(minutes=mins) if pd.notna(mins) else pd.NaT
    )
    agg_df = agg_df[pd.notna(agg_df["PLANNED_PICKING_DEADLINE_TS"])].copy()

    # Clean types
    agg_df["PLANNED_FRAME_STACK_ID"] = agg_df["PLANNED_FRAME_STACK_ID"].astype(str)
    agg_df["SHIPMENT_ID"] = agg_df["SHIPMENT_ID"].astype(str)
    agg_df["ACTUAL_START_TS"] = pd.to_datetime(agg_df["ACTUAL_START_TS"])
    agg_df["ACTUAL_END_TS"] = pd.to_datetime(agg_df["ACTUAL_END_TS"])

    # Sort shipment IDs by earliest deadline (for consistent color order)
    shipment_deadline_df = agg_df.groupby("SHIPMENT_ID")["PLANNED_PICKING_DEADLINE_TS"].min().reset_index()
    shipment_deadline_df = shipment_deadline_df.sort_values("PLANNED_PICKING_DEADLINE_TS")
    ordered_shipments = shipment_deadline_df["SHIPMENT_ID"].tolist()

    # Sort frame-stacks by descending deadline (to appear top-down)
    ordering_df = agg_df[["SHIPMENT_ID", "PLANNED_FRAME_STACK_ID", "PLANNED_PICKING_DEADLINE_TS"]].drop_duplicates()
    ordering_df = ordering_df.sort_values("PLANNED_PICKING_DEADLINE_TS", ascending=False)
    ordered_framestacks = ordering_df["PLANNED_FRAME_STACK_ID"].tolist()

    # Create mapping from frame-stack to visual y-position
    cat_to_rev_y = {cat: len(ordered_framestacks) - 1 - i for i, cat in enumerate(ordered_framestacks)}

    # Plot
    fig = px.timeline(
        agg_df,
        x_start="ACTUAL_START_TS",
        x_end="ACTUAL_END_TS",
        y="PLANNED_FRAME_STACK_ID",
        color="SHIPMENT_ID",
        hover_data=["PLANNED_PICKING_DEADLINE"],
        category_orders={
            "PLANNED_FRAME_STACK_ID": ordered_framestacks,
            "SHIPMENT_ID": ordered_shipments
        },
        title="Frame-Stack Picking Timeline (Jan 24, 2025, AMBIENT — Earlier Deadlines at Top)"
    )
    fig.update_yaxes(autorange="reversed", showticklabels=False)

    # Add deadline lines at proper height
    for _, row in agg_df.iterrows():
        fs = row["PLANNED_FRAME_STACK_ID"]
        deadline = row["PLANNED_PICKING_DEADLINE_TS"]
        if pd.isna(deadline) or fs not in cat_to_rev_y:
            continue
        y = cat_to_rev_y[fs]
        fig.add_shape(
            type="line",
            x0=deadline, x1=deadline,
            y0=y - 0.4, y1=y + 0.4,
            xref="x", yref="y",
            line=dict(color="black", width=2)
        )

    # Save HTML
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"✅ Visualization saved to: {output_path}")


# Run visualization
visualize_frame_stack_schedule(agg_df)
