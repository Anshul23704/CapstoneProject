from queue import Queue
import sys
from stage1_frame_ingestion import FrameIngestionStage, FrameIngestionConfig
from stage2_detection_tracking import DetectionTrackingStage, DetectionConfig
from stage3_active_buffering import ActiveBufferingStage, BufferingConfig
from stage4_vehicle_finalization import VehicleFinalizationStage
from stage5_job_creation import JobCreationStage
from stage6_worker_pool import WorkerPoolStage


processing_queue = Queue()
result_queue = Queue()

ingestion = FrameIngestionStage(FrameIngestionConfig(source="traffic.mp4"))
detection = DetectionTrackingStage(DetectionConfig())
buffering = ActiveBufferingStage(BufferingConfig())
finalization = VehicleFinalizationStage()
job_creator = JobCreationStage(processing_queue)
worker_pool = WorkerPoolStage(processing_queue, result_queue)

worker_pool.start()

with ingestion:
    print("Starting pipeline...\n")
    total_frames = int(ingestion._cap.get(7))
    for frame, frame_idx, ts in ingestion.frames():
        sys.stdout.write(f"\rFrame {frame_idx+1} / {total_frames}")
        sys.stdout.flush()

        track_map = detection.process(frame)

        ready_buffers = buffering.update(track_map, frame, frame_idx, ts)

        for buf in ready_buffers:

            fv = finalization.process(buf)

            if fv:
                job_creator.dispatch(fv)

        while not result_queue.empty():
            result = result_queue.get()
            print("Plate:", result.plate_text)