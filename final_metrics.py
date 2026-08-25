import os
import sqlite3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a publication-ready styling
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

def save_fig(fig, path, name):
    """Save figure as both PNG and PDF for academic use."""
    fig.savefig(os.path.join(path, f"{name}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(path, f"{name}.pdf"), dpi=300, bbox_inches='tight')

def generate_metrics_report(run_dir, processing_time=None, total_frames=None, fps=30.0):
    db_path = os.path.join(run_dir, "results.db")
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        return

    # Load data from SQLite
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM ocr_results", conn)
    conn.close()

    if df.empty:
        print("Error: No data found in ocr_results table.")
        return

    metrics = {}

    # 1. Pipeline Funnel Metrics (Yield)
    total_vehicles = df['track_id'].nunique()
    vehicles_with_readings = df[df['num_readings'] > 0]['track_id'].nunique()
    
    # is_valid indicates if the temporal fusion yielded a valid plate
    success_df = df[(df['is_valid'] == 1) & (df['plate_text'] != '')].copy()
    vehicles_successful_ocr = success_df['track_id'].nunique()
    
    metrics['total_tracks'] = total_vehicles
    metrics['tracks_with_crops'] = vehicles_with_readings
    metrics['successful_reads'] = vehicles_successful_ocr
    metrics['system_retention_rate'] = (vehicles_successful_ocr / total_vehicles) * 100 if total_vehicles > 0 else 0

    # 2. High-Confidence Yield
    high_conf_threshold = 0.85
    high_conf_reads = success_df[success_df['confidence'] >= high_conf_threshold].shape[0]
    metrics['high_conf_yield'] = (high_conf_reads / vehicles_successful_ocr) * 100 if vehicles_successful_ocr > 0 else 0
    metrics['avg_confidence'] = success_df['confidence'].mean() if not success_df.empty else 0

    # 3. Track Fragmentation Index
    # Stitched tracks mean a single physical vehicle was fragmented into multiple Track IDs
    stitched_count = 0
    if 'stitched_track_ids' in df.columns:
        # Count tracks that have a non-empty stitched_track_ids string
        stitched_count = df[df['stitched_track_ids'] != ''].shape[0]
    metrics['stitched_tracks'] = stitched_count
    metrics['fragmentation_index'] = (stitched_count / total_vehicles) * 100 if total_vehicles > 0 else 0

    # 4. Processing Efficiency & RTF
    if processing_time is not None and total_frames is not None and total_frames > 0:
        video_duration = total_frames / fps
        rtf = processing_time / video_duration
        processing_fps = total_frames / processing_time
        metrics['processing_time'] = processing_time
        metrics['video_duration'] = video_duration
        metrics['rtf'] = rtf
        metrics['processing_fps'] = processing_fps
    else:
        metrics['rtf'] = None

    # 5. Preprocessing Efficacy (Ablation Proxy)
    winner_counts = {}
    if not success_df.empty and 'winner_branch' in success_df.columns:
        winner_counts = success_df['winner_branch'].value_counts().to_dict()

    # ── GENERATE GRAPHS ──

    # Graph A: Pipeline Funnel (Bar Chart)
    fig_funnel, ax = plt.subplots(figsize=(8, 5))
    funnel_stages = ['Total Tracks', 'Tracks w/ BBox', 'Validated Plate']
    funnel_values = [total_vehicles, vehicles_with_readings, vehicles_successful_ocr]
    sns.barplot(x=funnel_stages, y=funnel_values, hue=funnel_stages, palette="Blues_d", legend=False, ax=ax)
    ax.set_title('ALPR Pipeline Vehicle Retention Funnel')
    ax.set_ylabel('Number of Vehicles')
    for i, v in enumerate(funnel_values):
        ax.text(i, v + (max(funnel_values)*0.02), str(v), ha='center', fontweight='bold')
    save_fig(fig_funnel, run_dir, "research_metric_funnel")
    plt.close(fig_funnel)

    # Graph B: Confidence Distribution (KDE + Histogram)
    fig_conf, ax = plt.subplots(figsize=(8, 5))
    if not success_df.empty:
        sns.histplot(success_df['confidence'], bins=15, kde=True, color='#2c3e50', ax=ax)
        ax.axvline(metrics['avg_confidence'], color='#e74c3c', linestyle='--', label=f'Mean = {metrics["avg_confidence"]:.2f}')
        ax.set_title('Distribution of Final OCR Confidence Scores')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.legend()
    save_fig(fig_conf, run_dir, "research_metric_confidence_dist")
    plt.close(fig_conf)

    # Graph C: Preprocessing Efficacy (Pie/Bar)
    if winner_counts:
        fig_branch, ax = plt.subplots(figsize=(7, 5))
        labels = list(winner_counts.keys())
        sizes = list(winner_counts.values())
        colors = ['#3498db', '#2ecc71', '#f39c12', '#95a5a6']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops={'edgecolor': 'w'})
        ax.set_title('Preprocessing Branch Efficacy (Winner Share)')
        save_fig(fig_branch, run_dir, "research_metric_preprocessing_efficacy")
        plt.close(fig_branch)

    # Graph D: Track Duration vs Confidence
    fig_dur, ax = plt.subplots(figsize=(8, 5))
    if not success_df.empty:
        sns.scatterplot(data=success_df, x='num_readings', y='confidence', alpha=0.7, color='#8e44ad', ax=ax)
        ax.set_title('Track Duration (Frames) vs. Fused Plate Confidence')
        ax.set_xlabel('Number of Fused Frames')
        ax.set_ylabel('Confidence')
    save_fig(fig_dur, run_dir, "research_metric_duration_vs_confidence")
    plt.close(fig_dur)

    # ── GENERATE MARKDOWN REPORT ──
    report_lines = [
        f"# Research-Grade System Metrics for `{os.path.basename(run_dir)}`",
        "",
        "## 1. System Throughput & Latency",
    ]
    if metrics['rtf'] is not None:
        report_lines.extend([
            f"- **Total Processing Time:** {metrics['processing_time']:.2f} seconds",
            f"- **Video Duration:** {metrics['video_duration']:.2f} seconds",
            f"- **Processing Speed:** {metrics['processing_fps']:.2f} FPS",
            f"- **Real-Time Factor (RTF):** {metrics['rtf']:.3f} *(< 1.0 indicates faster than real-time)*"
        ])
    else:
        report_lines.append("- *Throughput metrics unavailable (pipeline timing not provided).*")

    report_lines.extend([
        "",
        "## 2. Detection & Recognition Yield (Funnel)",
        f"- **Total Tracks Initiated (Stage 2):** {metrics['total_tracks']}",
        f"- **Tracks with Valid ROI Crops (Stage 3):** {metrics['tracks_with_crops']}",
        f"- **Tracks with Validated OCR Plate (Stage 7):** {metrics['successful_reads']}",
        f"- **Overall Pipeline Retention Rate:** {metrics['system_retention_rate']:.1f}%",
        "",
        "## 3. Recognition Quality & Confidence",
        f"- **Average Fused Confidence:** {metrics['avg_confidence']:.4f}",
        f"- **High-Confidence Yield (>85%):** {metrics['high_conf_yield']:.1f}% of successful reads",
        "",
        "## 4. Tracking Integrity (Fragmentation)",
        f"- **Stitched Tracks (ID Switches Repaired):** {metrics['stitched_tracks']}",
        f"- **Track Fragmentation Index:** {metrics['fragmentation_index']:.1f}% *(lower is better)*",
        "",
    ])

    if winner_counts:
        report_lines.extend([
            "## 5. Preprocessing Efficacy",
            "- **Winning Branches for Final Fused Output:**",
        ])
        for branch, count in winner_counts.items():
            report_lines.append(f"  - **{branch.capitalize()}:** {count} plates ({(count/metrics['successful_reads'])*100:.1f}%)")
        report_lines.append("")

    report_lines.extend([
        "## Generated Research Graphs (Available in PNG and PDF)",
        "- `research_metric_funnel.pdf` / `.png`",
        "- `research_metric_confidence_dist.pdf` / `.png`",
        "- `research_metric_preprocessing_efficacy.pdf` / `.png`",
        "- `research_metric_duration_vs_confidence.pdf` / `.png`"
    ])

    report_path = os.path.join(run_dir, "research_system_metrics.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[final_metrics] Research metrics generated in: {run_dir}")
    print(f"[final_metrics] Report: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate research-grade pipeline validation metrics.")
    parser.add_argument("run_dir", type=str, help="Path to the output run directory")
    args = parser.parse_args()
    
    # When run directly from CLI without timing data
    generate_metrics_report(args.run_dir)
