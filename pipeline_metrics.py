import os
import cv2
import numpy as np

class PipelineMetricsLogger:
    def __init__(self, run_id: str):
        self.run_id = run_id
        
        self.total_detections = 0
        self.plate_attempts = 0
        self.unique_plates_detected = 0
        self.ocr_success_count = 0
        
        self.finalized_vehicles = []
        
    @staticmethod
    def _sharpness(crop: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        if crop is None or crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
    def log_finalized_vehicle(self, fv):
        """Record metrics for a finalized vehicle."""
        max_sharpness = 0.0
        if hasattr(fv, 'frames') and fv.frames:
            for f in fv.frames:
                if hasattr(f, 'crop') and f.crop is not None:
                    s = self._sharpness(f.crop)
                    if s > max_sharpness:
                        max_sharpness = s
                        
        self.finalized_vehicles.append({
            'track_id': getattr(fv, 'track_id', -1),
            'duration': getattr(fv, 'track_duration', 0),
            'avg_bbox_area': getattr(fv, 'avg_bbox_area', 0.0),
            'finalize_reason': getattr(fv, 'finalize_reason', 'unknown'),
            'low_diversity': getattr(fv, 'low_diversity', False),
            'possible_id_switch': getattr(fv, 'possible_id_switch', False),
            'max_sharpness': max_sharpness
        })
        
    def log_detection(self, count=1):
        self.total_detections += count
        
    def log_plate_attempt(self):
        self.plate_attempts += 1
        
    def generate_report(self, output_dir: str, ingestion_stage):
        """Generate a markdown report with the collected metrics."""
        report_lines = []
        report_lines.append(f"# ALPR Pipeline Metrics - Run {self.run_id}\n")
        
        # Stage 1
        report_lines.append("## Stage 1: Frame Ingestion")
        report_lines.append(f"- **Total frames processed:** {getattr(ingestion_stage, 'total_frames', 0)}")
        report_lines.append(f"- **Dropped frames:** {getattr(ingestion_stage, 'dropped_frames', 0)}")
        
        actual_fps = getattr(ingestion_stage, 'actual_fps', 0.0)
        report_lines.append(f"- **Actual FPS:** {actual_fps:.2f}\n")
        
        # Stage 2
        report_lines.append("## Stage 2: Detection & Tracking")
        report_lines.append(f"- **Total vehicle detections across all frames:** {self.total_detections}")
        report_lines.append(f"- **Unique vehicles finalized:** {len(self.finalized_vehicles)}\n")
        
        # Stages 3 & 4
        report_lines.append("## Stage 3 & 4: Buffering & Finalization")
        report_lines.append("| Track ID | Duration | Avg BBox Area | Finalize Reason | Low Diversity | Possible ID Switch | Max Sharpness |")
        report_lines.append("|----------|----------|---------------|-----------------|---------------|--------------------|---------------|")
        for v in self.finalized_vehicles:
            report_lines.append(
                f"| {v['track_id']} | {v['duration']} | {v['avg_bbox_area']:.1f} | {v['finalize_reason']} | "
                f"{v['low_diversity']} | {v['possible_id_switch']} | {v['max_sharpness']:.2f} |"
            )
        report_lines.append("\n")
        
        # Stages 5 & 6
        report_lines.append("## Stage 5 & 6: OCR (Inline Processing)")
        report_lines.append(f"- **Plate crops attempted:** {self.plate_attempts}")
        report_lines.append(f"- **Successful OCR reads:** {self.ocr_success_count}")
        report_lines.append(f"- **Unique plates detected:** {self.unique_plates_detected}\n")
        
        # Write to file
        out_path = os.path.join(output_dir, "pipeline_metrics.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
            
        return out_path
