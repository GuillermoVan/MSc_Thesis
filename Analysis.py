import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class FrameStackAnalyzer:
    def __init__(self, csv_path: str, completion_time_cap: float = 600):
        self.csv_path = csv_path
        self.completion_time_cap = completion_time_cap
        self.df_start = None

    def load_and_prepare_data(self):
        df = pd.read_csv(self.csv_path)
        df_start = df[df['PLANNED_TOTE_ID'] == 'START'].copy()
        df_start['START_PICKING_TS'] = pd.to_datetime(df_start['START_PICKING_TS'], errors='coerce')
        df_start = df_start[df_start['REMAINING_PICKING_TIME'].notna()]
        df_start = df_start[
            (df_start['REMAINING_PICKING_TIME'] > 0) &
            (df_start['REMAINING_PICKING_TIME'] <= self.completion_time_cap)
        ]
        self.df_start = df_start

    def _get_summary_table(self, full, morning, afternoon):
        def group_stats(df, label):
            return {
                f'{label} - All': df['REMAINING_PICKING_TIME'].mean(),
                f'{label} - AMBIENT': df[df['TEMPZONE'] == 'AMBIENT']['REMAINING_PICKING_TIME'].mean(),
                f'{label} - CHILLED': df[df['TEMPZONE'] == 'CHILLED']['REMAINING_PICKING_TIME'].mean(),
            }

        stats = {}
        stats.update(group_stats(full, "Overall"))
        stats.update(group_stats(morning, "Morning"))
        stats.update(group_stats(afternoon, "Afternoon"))

        rows = []
        for label in ["Overall - All", "Overall - AMBIENT", "Overall - CHILLED",
                      "Morning - All", "Morning - AMBIENT", "Morning - CHILLED",
                      "Afternoon - All", "Afternoon - AMBIENT", "Afternoon - CHILLED"]:
            value = stats[label]
            rows.append([label, f"{value:.1f}" if pd.notna(value) else "N/A"])
        return rows

    def plot_combined_view(self, cutoff_hour: int = 13):
        if self.df_start is None:
            raise ValueError("Data not loaded. Run load_and_prepare_data() first.")

        full = self.df_start
        morning = full[full['START_PICKING_TS'].dt.hour < cutoff_hour]
        afternoon = full[full['START_PICKING_TS'].dt.hour >= cutoff_hour]

        full['DATE'] = full['START_PICKING_TS'].dt.date
        full['TIME_OF_DAY'] = full['START_PICKING_TS'].dt.hour < cutoff_hour

        spread_df = full.groupby(['DATE', 'TIME_OF_DAY'])['REMAINING_PICKING_TIME'].std().reset_index()
        spread_df['TIME_OF_DAY'] = spread_df['TIME_OF_DAY'].map({True: 'Morning', False: 'Afternoon'})

        # Set up 2x3 grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # 1. All frame-stacks
        sns.histplot(full['REMAINING_PICKING_TIME'], bins=40, kde=True,
                    color='mediumseagreen', edgecolor='black', ax=axes[0])
        axes[0].set_title("All frame-stacks", fontsize=12)
        axes[0].set_xlabel("Completion Time [min]")
        axes[0].set_ylabel("# Frame-Stacks")
        axes[0].grid(True)

        # 2. Morning
        sns.histplot(morning['REMAINING_PICKING_TIME'], bins=40, kde=True,
                    color='orange', edgecolor='black', ax=axes[1])
        axes[1].set_title(f"Picking start < {cutoff_hour}:00", fontsize=12)
        axes[1].set_xlabel("Completion Time [min]")
        axes[1].set_ylabel("# Frame-Stacks")
        axes[1].grid(True)

        # 3. Afternoon
        sns.histplot(afternoon['REMAINING_PICKING_TIME'], bins=40, kde=True,
                    color='skyblue', edgecolor='black', ax=axes[2])
        axes[2].set_title(f"Picking start >= {cutoff_hour}:00", fontsize=12)
        axes[2].set_xlabel("Completion Time [min]")
        axes[2].set_ylabel("# Frame-Stacks")
        axes[2].grid(True)

        # 4. Summary table
        axes[3].axis('off')
        table_data = self._get_summary_table(full, morning, afternoon)
        col_labels = ["Group", "Avg Completion Time (min)"]
        table = axes[3].table(cellText=table_data, colLabels=col_labels,
                            cellLoc='center', loc='center')
        table.scale(1, 2.2)
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        axes[3].text(0.5, 1.1, "Summary Statistics", ha='center', va='bottom', fontsize=14, transform=axes[3].transAxes)

        # 5. Boxplot of daily std dev
        sns.boxplot(data=spread_df, x='TIME_OF_DAY', y='REMAINING_PICKING_TIME',
                    palette='Set2', ax=axes[4])
        axes[4].set_title("Spread of Daily Std. Devs per Time of Day", fontsize=12)
        axes[4].set_ylabel("Daily Std Dev of Completion Time [min]")
        axes[4].set_xlabel("Time of Day")

        # 6. Daily mean
        daily_mean = full.groupby('DATE')['REMAINING_PICKING_TIME'].mean().reset_index()
        axes[5].plot(daily_mean['DATE'], daily_mean['REMAINING_PICKING_TIME'], marker='o')
        axes[5].set_title("Average Completion Time per Day", fontsize=12)
        axes[5].set_xlabel("Date")
        axes[5].set_ylabel("Avg Completion Time [min]")
        axes[5].tick_params(axis='x', labelrotation=45)

        # Layout adjustments
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        fig.suptitle(f"Frame-Stack Completion Time Distributions (≤ {self.completion_time_cap} min)",
                    fontsize=18, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()



# Example usage
if __name__ == "__main__":
    analyzer = FrameStackAnalyzer(csv_path="Data/Training/ACTIVE_features_4_months.csv", 
                                  completion_time_cap=600)
    analyzer.load_and_prepare_data()
    analyzer.plot_combined_view(cutoff_hour=13)
