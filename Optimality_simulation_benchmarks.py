import pandas as pd
from gurobipy import Model, GRB, quicksum
import numpy as np
import os
import plotly.express as px


class Benchmarking:
    def __init__(
        self,
        gbdt_data_path: str | None,
        normal_data_path: str | None,
        start_time: str,
        end_time: str,
        time_step_minutes: int = 3,
        capacity: int = 70,
        edd_fs_prio_weight: float = 1.0,
        min_start_gap_minutes: int = 3,
        tempzone: str = 'Both',
        nr_of_concurrent_starting_fs: int = 2,
        save_path: str = 'Results_benchmark'
        ):

        self.start_time = start_time
        self.end_time = end_time
        self.time_step_minutes = time_step_minutes
        self.capacity = capacity
        self.edd_fs_prio_weight = edd_fs_prio_weight
        self.min_start_gap_minutes = min_start_gap_minutes
        self.tempzone = tempzone
        self.nr_of_concurrent_starting_fs = nr_of_concurrent_starting_fs
        self.save_path = save_path

        if gbdt_data_path is not None:
            self.gbdt_schedule_df = self.csv_to_df(gbdt_data_path)
        else:
            self.gbdt_schedule_df = None

        if normal_data_path is not None:
            self.normal_schedule_df = self.csv_to_df(normal_data_path)
        else:
            self.normal_schedule_df = None
    
    
    def csv_to_df(self, csv_path):
        "Converts the data in the CSV file to a df (with columns for start_time, end_time and deadline per framestack) and stores this as an attribute"
        
        df = pd.read_csv(csv_path, sep=";")
        df['ACTUAL_START_TS_SIM'] = pd.to_datetime(df['ACTUAL_START_TS_SIM'], errors='coerce')
        df['ACTUAL_END_TS_SIM'] = pd.to_datetime(df['ACTUAL_END_TS_SIM'], errors='coerce')
        df['PLANNED_PICKING_DEADLINE_TS'] = pd.to_datetime(df['PLANNED_PICKING_DEADLINE_TS'])
        df['PLANNED_DEPARTURE_DEADLINE_TS'] = pd.to_datetime(df['PLANNED_DEPARTURE_DEADLINE_TS'])
        df = df[~(df['ACTUAL_START_TS_SIM'].isna() & df['ACTUAL_END_TS_SIM'].isna())]

        return df
    

    def calculate_optimality(self, schedule_df):
        if schedule_df is None or schedule_df.empty:
            #print('No input data found')
            return None

        df = schedule_df.copy()

        t0_minutes = pd.to_datetime(self.start_time).timestamp() / 60
        df['deadline_minutes'] = df['PLANNED_PICKING_DEADLINE_TS'].apply(lambda ts: ts.timestamp() / 60)
        df['end_time_minutes'] = df['ACTUAL_END_TS_SIM'].apply(lambda ts: ts.timestamp() / 60)

        df['T_i'] = df['PLANNED_PICKING_DEADLINE_TS'] - df['ACTUAL_END_TS_SIM']
        df['w_i'] = self.edd_fs_prio_weight / (df['deadline_minutes'] - t0_minutes).clip(lower=1e-6)
        df['contribution'] = df['w_i'] * df['T_i']
        total_objective = df['contribution'].sum().total_seconds() / 60

        return total_objective
    

    def generate_hindsight_optimal_schedule(self, schedule_df):
        if schedule_df is None or schedule_df.empty:
            #print('No input data found')
            return None

        df = schedule_df.copy()
        df = df.dropna(subset=['ACTUAL_END_TS_SIM', 'ACTUAL_START_TS_SIM'])

        df['duration_min'] = (df['ACTUAL_END_TS_SIM'] - df['ACTUAL_START_TS_SIM']).dt.total_seconds() / 60
        df['P_i'] = np.ceil(df['duration_min'] / self.time_step_minutes).astype(int)

        t0 = pd.to_datetime(self.start_time)
        T_max = int((pd.to_datetime(self.end_time) - t0).total_seconds() // 60 // self.time_step_minutes)
        slots = range(T_max + 1)
        frame_stacks = list(df['PLANNED_FRAME_STACK_ID'])
        df = df.set_index('PLANNED_FRAME_STACK_ID')

        m = Model("HindsightOptimal")
        m.setParam("OutputFlag", 0)

        x = m.addVars(frame_stacks, slots, vtype=GRB.BINARY, name="x")

        for i in frame_stacks:
            m.addConstr(quicksum(x[i, t] for t in slots) == 1)

        for t in slots:
            m.addConstr(
                quicksum(
                    x[i, s]
                    for i in frame_stacks for s in slots
                    if s <= t < s + df.at[i, 'P_i']
                ) <= self.capacity
            )

        for t in slots:
            m.addConstr(
                quicksum(x[i, t] for i in frame_stacks) <= self.nr_of_concurrent_starting_fs
            )

        t0_minutes = t0.timestamp() / 60
        w = {
            i: self.edd_fs_prio_weight / max(1e-6, (df.at[i, 'PLANNED_PICKING_DEADLINE_TS'].timestamp() / 60) - t0_minutes)
            for i in frame_stacks
        }

        objective = quicksum(
            w[i] * (
                df.at[i, 'PLANNED_PICKING_DEADLINE_TS'].timestamp() / 60 -
                quicksum(((t0_minutes + t * self.time_step_minutes + df.at[i, 'duration_min']) * x[i, t]) for t in slots)
            )
            for i in frame_stacks
        )

        m.setObjective(objective, GRB.MAXIMIZE)
        m.optimize()

        records = []
        for i in frame_stacks:
            for t in slots:
                if x[i, t].X > 0.5:
                    start = t0 + pd.Timedelta(minutes=t * self.time_step_minutes)
                    end = start + pd.Timedelta(minutes=df.at[i, 'duration_min'])
                    records.append({
                        'PLANNED_FRAME_STACK_ID': i,
                        'KEY_TRUCK_TRIP': df.at[i, 'KEY_TRUCK_TRIP'],  # <== include shipment ID
                        'ACTUAL_START_TS_SIM': start,
                        'ACTUAL_END_TS_SIM': end,
                        'PLANNED_PICKING_DEADLINE_TS': df.at[i, 'PLANNED_PICKING_DEADLINE_TS'],
                        'PLANNED_DEPARTURE_DEADLINE_TS': df.at[i, 'PLANNED_DEPARTURE_DEADLINE_TS']
                    })

        return pd.DataFrame(records)

    
    def visualize_schedule(self, schedule_df, title):
        "Creates an HTML file with the schedule in Gantt format and saves it in the save_path"
        if schedule_df is None:
            return

        os.makedirs(self.save_path, exist_ok=True)
        html_path = os.path.join(self.save_path, f"{title.replace(' ', '_')}.html")

        df = schedule_df.copy()
        df = df.sort_values(by='ACTUAL_START_TS_SIM')

        df['shipment_id'] = df.get('shipment_id', df.get('KEY_TRUCK_TRIP', 'UNKNOWN'))
        df['ACTUAL_START_TS_SIM'] = pd.to_datetime(df['ACTUAL_START_TS_SIM'])
        df['ACTUAL_END_TS_SIM'] = pd.to_datetime(df['ACTUAL_END_TS_SIM'])
        df['PLANNED_PICKING_DEADLINE_TS'] = pd.to_datetime(df['PLANNED_PICKING_DEADLINE_TS'])

        # Sort shipment IDs by earliest deadline (for consistent color order)
        shipment_deadline_df = df.groupby('shipment_id')['PLANNED_PICKING_DEADLINE_TS'].min().reset_index()
        shipment_deadline_df = shipment_deadline_df.sort_values(by='PLANNED_PICKING_DEADLINE_TS')
        ordered_shipments = shipment_deadline_df['shipment_id'].tolist()

        # Sort frame-stacks for y-axis (bottom = earliest deadline)
        ordering_df = df[['shipment_id', 'PLANNED_FRAME_STACK_ID', 'PLANNED_PICKING_DEADLINE_TS']].drop_duplicates()
        ordering_df = ordering_df.sort_values(by='PLANNED_PICKING_DEADLINE_TS', ascending=False)
        ordered_framestacks = ordering_df['PLANNED_FRAME_STACK_ID'].tolist()
        cat_to_rev_y = {cat: len(ordered_framestacks) - 1 - i for i, cat in enumerate(ordered_framestacks)}

        fig = px.timeline(
            df,
            x_start='ACTUAL_START_TS_SIM', x_end='ACTUAL_END_TS_SIM',
            y='PLANNED_FRAME_STACK_ID', color='shipment_id',
            hover_data=['PLANNED_FRAME_STACK_ID'],
            category_orders={
                'PLANNED_FRAME_STACK_ID': ordered_framestacks,
                'shipment_id': ordered_shipments
            },
            title=title
        )
        fig.update_yaxes(autorange='reversed', showticklabels=False)

        for _, row in df.iterrows():
            fs = row['PLANNED_FRAME_STACK_ID']
            picking_dl = row['PLANNED_PICKING_DEADLINE_TS']
            if fs not in cat_to_rev_y or pd.isna(picking_dl):
                continue
            y = cat_to_rev_y[fs]
            fig.add_shape(
                type='line',
                x0=picking_dl.to_pydatetime(), x1=picking_dl.to_pydatetime(),
                y0=y - 0.4, y1=y + 0.4,
                xref='x', yref='y',
                line=dict(color='black', width=2)
            )

        fig.write_html(html_path, include_plotlyjs='cdn')

        self._save_schedule_as_csv(df, title)

    
    def _save_schedule_as_csv(self, schedule_df, title):
        if schedule_df is None:
            return

        os.makedirs(self.save_path, exist_ok=True)
        filename = f"{title.replace(' ', '_')}.csv"
        csv_path = os.path.join(self.save_path, filename)

        schedule_df.to_csv(csv_path, index=False, sep=';')
    
    def calculate_additional_metrics(self, schedule_df):
        if schedule_df is None or schedule_df.empty:
            return None, None, None

        df = schedule_df.copy()

        # Average makespan per shipment
        df['start_min'] = df['ACTUAL_START_TS_SIM']
        df['end_min'] = df['ACTUAL_END_TS_SIM']
        makespan_per_shipment = df.groupby('KEY_TRUCK_TRIP').agg(
            shipment_start=('start_min', 'min'),
            shipment_end=('end_min', 'max')
        )
        makespan_per_shipment['makespan_min'] = (
            (makespan_per_shipment['shipment_end'] - makespan_per_shipment['shipment_start']).dt.total_seconds() / 60
        )
        avg_makespan = makespan_per_shipment['makespan_min'].mean()

        # Total delay (minutes)
        df['delay_min'] = (df['ACTUAL_END_TS_SIM'] - df['PLANNED_PICKING_DEADLINE_TS']).dt.total_seconds() / 60
        df['delay_min'] = df['delay_min'].clip(lower=0)  # only count positive delays
        total_delay = df['delay_min'].sum()

        # Delayed frame-stack count
        delayed_fs_count = (df['delay_min'] > 0).sum()

        return avg_makespan, total_delay, delayed_fs_count


if __name__ == '__main__':
    benchmark = Benchmarking(
                        gbdt_data_path = "Simulation_output\Simulation schedule output 24 jan.csv",
                        normal_data_path = None,
                        start_time = "2025-01-24 04:30:00",
                        end_time = "2025-01-24 23:00:00",
                        time_step_minutes = 3,
                        capacity = 70,
                        edd_fs_prio_weight = 1.0,
                        min_start_gap_minutes = 3,
                        tempzone = 'Ambient', # Only ambient is available in the simulation
                        nr_of_concurrent_starting_fs = 2,
                        save_path = 'Results_benchmark'
                        )
    
    # Generate in hindsight optimal schedule
    gbdt_hindsight_optimal_schedule = benchmark.generate_hindsight_optimal_schedule(benchmark.gbdt_schedule_df)
    normal_hindsight_optimal_schedule = benchmark.generate_hindsight_optimal_schedule(benchmark.normal_schedule_df)


    # Assess optimality per method
    optimality_gbdt = benchmark.calculate_optimality(benchmark.gbdt_schedule_df)
    print(f'Objective of schedule with GBDT = {optimality_gbdt}')

    optimality_normal = benchmark.calculate_optimality(benchmark.normal_schedule_df)
    print(f'Objective of schedule normal = {optimality_normal}')

    optimality_gbdt_hindsight_optimal = benchmark.calculate_optimality(gbdt_hindsight_optimal_schedule)
    print(f'Objective of hindsight optimal schedule based on GBDT integrated schedule = {optimality_gbdt_hindsight_optimal}')

    optimality_normal_hindsight_optimal = benchmark.calculate_optimality(normal_hindsight_optimal_schedule)
    print(f'Objective of hindsight optimal schedule based on normal operations = {optimality_normal_hindsight_optimal}')


    # Visualizations
    benchmark.visualize_schedule(benchmark.gbdt_schedule_df, title="Result with GBDT model integrated")
    benchmark.visualize_schedule(benchmark.normal_schedule_df, title="Result without GBDT model integrated")
    benchmark.visualize_schedule(gbdt_hindsight_optimal_schedule, title="Result with hindsight optimum based on GBDT integrated operations")
    benchmark.visualize_schedule(normal_hindsight_optimal_schedule, title="Result with hindsight optimum based on normal operations")

    # Additional metrics for GBDT model integrated
    avg_makespan_gbdt, total_delay_gbdt, delayed_fs_gbdt = benchmark.calculate_additional_metrics(benchmark.gbdt_schedule_df)
    print(f'Average makespan per shipment (GBDT) = {avg_makespan_gbdt:.2f} min')
    print(f'Total delay (GBDT) = {total_delay_gbdt:.2f} min')
    print(f'Delayed frame-stacks (GBDT) = {delayed_fs_gbdt}')

    # Additional metrics for hindsight optimum based on GBDT
    avg_makespan_gbdt_opt, total_delay_gbdt_opt, delayed_fs_gbdt_opt = benchmark.calculate_additional_metrics(gbdt_hindsight_optimal_schedule)
    print(f'Average makespan per shipment (GBDT hindsight) = {avg_makespan_gbdt_opt:.2f} min')
    print(f'Total delay (GBDT hindsight) = {total_delay_gbdt_opt:.2f} min')
    print(f'Delayed frame-stacks (GBDT hindsight) = {delayed_fs_gbdt_opt}')

