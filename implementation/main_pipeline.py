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

ingestion = FrameIngestionStage(FrameIngestionConfig(source="D:\\Sem6_Subjects\\Capstone\\implementation\\traffic.mp4"))
detection = DetectionTrackingStage(DetectionConfig())
buffering = ActiveBufferingStage(BufferingConfig())
finalization = VehicleFinalizationStage()
job_creator = JobCreationStage(processing_queue)
worker_pool = WorkerPoolStage(processing_queue, result_queue)

worker_pool.start()
frame_count = 0
detection_count = 0
buffer_count = 0
finalized_count = 0
job_count = 0
result_count = 0
with ingestion:
    total_frames = int(ingestion._cap.get(7))

    print("Starting pipeline...\n")

    for frame, frame_idx, ts in ingestion.frames():

        frame_count += 1

        track_map = detection.process(frame)
        detection_count += len(track_map)

        ready_buffers = buffering.update(track_map, frame, frame_idx, ts)
        buffer_count += len(buffering._new_track_ids)

        for buf in ready_buffers:
            fv = finalization.process(buf)
            if fv:
                finalized_count += 1
                job = job_creator.dispatch(fv)
                if job:
                    job_count += 1

       
        while not result_queue.empty():
            result = result_queue.get()
            result_count += 1

      
        sys.stdout.write(
            f"\rFrame {frame_idx+1}/{total_frames} | "
            f"Det:{detection_count} | Buf:{buffer_count} | "
            f"Fin:{finalized_count} | Jobs:{job_count} | Res:{result_count}"
        )
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
