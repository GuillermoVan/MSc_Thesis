import os
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import plotly.express as px
from datetime import datetime
from typing import Dict, Union, Tuple

# A completion value can be a flat duration (minutes) or a tuple (status, minutes)
CompletionValue = Union[float, Tuple[str, float]]

class PickingScheduler:
    def __init__(
        self,
        data_path: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        completions_dict: Dict[str, CompletionValue],
        time_step_minutes: int = 3, # This should be equal to min_start_gap_minutes for minimum runtime
        capacity: int = 70,
        edd_fs_prio_weight: float = 1.0,
        min_start_gap_minutes: int = 3,  # NEW
        tempzone: str = 'Both'
    ):
        # Input file and time window; accept str or datetime
        self.data_path = data_path
        self.start_time = start_time if isinstance(start_time, datetime) else pd.to_datetime(start_time)
        self.end_time = end_time if isinstance(end_time, datetime) else pd.to_datetime(end_time)

        # Completion times provided by user (in minutes) or with status
        if not completions_dict:
            raise ValueError("completions_dict must be provided and non-empty.")
        self.completions_dict = completions_dict

        # Temperature zone filter: 'Both', 'Ambient', or 'Chilled'
        valid_zones = {'Both', 'Ambient', 'Chilled'}
        if tempzone not in valid_zones:
            raise ValueError(f"tempzone must be one of {valid_zones}, got '{tempzone}'.")
        self.tempzone = tempzone

        # Discretization & capacity
        self.time_step = time_step_minutes
        self.capacity = capacity
        self.edd_fs_prio_weight = edd_fs_prio_weight
        self.min_start_gap = min_start_gap_minutes

        # Placeholders for data, model, and results
        self.raw = None
        self.model = None
        self.schedule_df = None

        # Determine Results directory relative to this file
        base_dir = os.path.join(os.path.dirname(__file__), 'Results_gbdt')
        os.makedirs(base_dir, exist_ok=True)
        self._results_dir = base_dir

        # Input file and time window; accept str or datetime
        self.data_path = data_path
        self.start_time = start_time if isinstance(start_time, datetime) else pd.to_datetime(start_time)
        self.end_time = end_time if isinstance(end_time, datetime) else pd.to_datetime(end_time)

        # Completion times provided by user (in minutes) or with status
        if not completions_dict:
            raise ValueError("completions_dict must be provided and non-empty.")
        self.completions_dict = completions_dict

        # Temperature zone filter: 'Both', 'Ambient', or 'Chilled'
        valid_zones = {'Both', 'Ambient', 'Chilled'}
        if tempzone not in valid_zones:
            raise ValueError(f"tempzone must be one of {valid_zones}, got '{tempzone}'.")
        self.tempzone = tempzone

    def load_and_preprocess(self):
        # Load data
        df = pd.read_csv(self.data_path)
        # Infer tempzone from frame_stack_id content
        fs = df['PLANNED_FRAME_STACK_ID'].astype(str).str.upper()
        df['TEMPZONE'] = np.where(
            fs.str.contains('COOLED'), 'Chilled',
            np.where(fs.str.contains('AMBIENT'), 'Ambient', None)
        )
        if self.tempzone != 'Both':
            df = df[df['TEMPZONE'] == self.tempzone]

        # Keep only required columns and filter stacks
        df['shipment_id'] = df['KEY_TRUCK_TRIP']
        df['frame_stack_id'] = df['PLANNED_FRAME_STACK_ID']
        df['shipment_deadline'] = pd.to_datetime(df['PLANNED_PICKING_DEADLINE_TS'])
        existing = set(df['frame_stack_id'])
        required = set(self.completions_dict.keys())
        missing = required - existing
        if missing:
            print(f"No deadlines found for frame-stacks: {list(missing)}")
        df = df[df['frame_stack_id'].isin(required)].copy()

        # Store the filtered raw data
        self.raw = df[['shipment_id', 'frame_stack_id', 'shipment_deadline']].copy()

        # Build time slots
        self.time_slots = pd.date_range(
            self.start_time, self.end_time, freq=f"{self.time_step}min"
        )
        self.slot_indices = list(range(len(self.time_slots)))

        self.raw['departure_deadline'] = pd.to_datetime(df['PLANNED_DEPARTURE_DEADLINE_TS'])

    def assign_durations_from_dict(self):
        # Map statuses and durations
        statuses = {}
        durations = {}
        for fs_id, val in self.completions_dict.items():
            if isinstance(val, tuple):
                status, mins = val
            else:
                status, mins = 'inactive', float(val)
            statuses[fs_id] = status
            durations[fs_id] = float(mins)

        self.raw['status'] = self.raw['frame_stack_id'].map(statuses)
        self.raw['pick_duration_minutes'] = self.raw['frame_stack_id'].map(durations)
        if self.raw['pick_duration_minutes'].isnull().any():
            missing = self.raw.loc[self.raw['pick_duration_minutes'].isnull(), 'frame_stack_id'].unique()
            raise KeyError(f"No completion time for frame-stack(s): {list(missing)}")
        # Convert minutes to slot counts
        self.raw['pick_duration_slots'] = (
            np.ceil(self.raw['pick_duration_minutes'] / self.time_step).astype(int)
        )

    def build_model(self):
        idx = self.raw.index
        slots = self.slot_indices
        m = gp.Model('Picking_Scheduling')

        # Decision variables
        x = m.addVars(idx, slots, vtype=GRB.BINARY, name='x')
        y = m.addVars(idx, vtype=GRB.CONTINUOUS, name='y')
        z = m.addVars(idx, vtype=GRB.CONTINUOUS, name='z')
        T = m.addVars(idx, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name='T')

        # Constraints for each frame-stack i
        for i in idx:
            m.addConstr(gp.quicksum(x[i, t] for t in slots) == 1, name=f'assign_{i}')
            m.addConstr(
                y[i] == gp.quicksum(
                    x[i, t] * (self.start_time.timestamp() / 60 + t * self.time_step)
                    for t in slots
                ), name=f'starttime_{i}'
            )
            m.addConstr(z[i] == y[i] + self.raw.at[i, 'pick_duration_minutes'], name=f'finishtime_{i}')
            dl = self.raw.at[i, 'shipment_deadline'].timestamp() / 60
            m.addConstr(T[i] == dl - z[i], name=f'slack_{i}')
            if self.raw.at[i, 'status'] == 'active':
                m.addConstr(x[i, 0] == 1, name=f'lock_active_{i}')

        # Capacity constraint
        ds = self.raw['pick_duration_slots']
        for t in slots:
            m.addConstr(
                gp.quicksum(
                    x[i, s]
                    for i in idx for s in slots
                    if s <= t < s + ds.at[i]
                ) <= self.capacity,
                name=f'capacity_{t}'
            )

        # Minimum start gap constraint
        inactive_idx = [i for i in idx if self.raw.at[i, 'status'] == 'inactive']

        for s in slots:
            m.addConstr(
                gp.quicksum(x[i, s] for i in inactive_idx) <= 1,
                name=f'start_limit_{s}'
            )

        # Objective
        t0 = self.start_time.timestamp() / 60
        w = {i: self.edd_fs_prio_weight / max(1e-6, (self.raw.at[i, 'shipment_deadline'].timestamp() / 60) - t0)
             for i in idx}
        m.setObjective(gp.quicksum(w[i] * T[i] for i in idx), GRB.MAXIMIZE)

        self.model = m
        self._x = x

    def solve(self, time_limit: int = 300):
        self.model.Params.TimeLimit = time_limit
        self.model.optimize()

    def extract_schedule(self):
        if self.model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            raise RuntimeError('No solution found')
        rec = []
        for i in self.raw.index:
            for t in self.slot_indices:
                if self._x[i, t].X > 0.5:
                    start_dt = self.start_time + pd.Timedelta(minutes=t * self.time_step)
                    end_dt = start_dt + pd.Timedelta(minutes=self.raw.at[i, 'pick_duration_minutes'])
                    rec.append({
                        'shipment_id': self.raw.at[i, 'shipment_id'],
                        'frame_stack_id': self.raw.at[i, 'frame_stack_id'],
                        'start_time': start_dt,
                        'end_time': end_dt
                    })
        return pd.DataFrame(rec)

    def save_outputs(self, df: pd.DataFrame, suffix: str):
        # Prepare output path
        html_path = os.path.join(self._results_dir, f'optimized_schedule_{suffix}.html')

        # Build the unique frame_stack ordering
        cats = list(dict.fromkeys(df['frame_stack_id']))
        n = len(cats)
        cat_to_rev_y = {cat: n - 1 - i for i, cat in enumerate(cats)}

        # Create the timeline plot
        fig = px.timeline(
            df,
            x_start='start_time', x_end='end_time',
            y='frame_stack_id', color='shipment_id',
            hover_data=['frame_stack_id'],
            category_orders={'frame_stack_id': cats},
            title=f'Picking Schedule ({suffix})'
        )
        fig.update_yaxes(autorange='reversed', showticklabels=False)

        # Collect deadlines per shipment & frame_stack
        dl_df = (
            self.raw[['shipment_id', 'frame_stack_id', 'shipment_deadline', 'departure_deadline']]
            .drop_duplicates()
            .sort_values('shipment_deadline', ascending=False)
        )
        groups = df.groupby('shipment_id')['frame_stack_id'].unique()

        for _, row in dl_df.iterrows():
            fs = row['frame_stack_id']
            sid = row['shipment_id']
            picking_dl = row['shipment_deadline']
            departure_dl = row['departure_deadline']
            if fs not in cat_to_rev_y:
                continue
            y = cat_to_rev_y[fs]

            # Picking deadline in black
            fig.add_shape(
                type='line',
                x0=picking_dl.to_pydatetime(), x1=picking_dl.to_pydatetime(),
                y0=y - 0.4, y1=y + 0.4,
                xref='x', yref='y',
                line=dict(color='black', width=2)
            )
            # Departure deadline in red
            fig.add_shape(
                type='line',
                x0=departure_dl.to_pydatetime(), x1=departure_dl.to_pydatetime(),
                y0=y - 0.4, y1=y + 0.4,
                xref='x', yref='y',
                line=dict(color='red', width=2, dash='dot')
            )

        # Save HTML
        fig.write_html(html_path, include_plotlyjs='cdn')



    def run(self):
        zones = ['Ambient', 'Chilled'] if self.tempzone == 'Both' else [self.tempzone]
        for zone in zones:
            self.tempzone = zone
            self.load_and_preprocess()
            self.assign_durations_from_dict()
            self.build_model()
            self.solve()
            df = self.extract_schedule()
            self.schedule_df = df
            suffix = zone.lower()
            self.save_outputs(df, suffix)