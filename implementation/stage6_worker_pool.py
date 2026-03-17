import threading
from queue import Queue
from dataclasses import dataclass
from enum import Enum, auto
import easyocr

class RecognitionStatus(Enum):
    SUCCESS = auto()
    FAILED = auto()

@dataclass
class RecognitionResult:
    job_id: str
    track_id: int
    plate_text: str
    confidence: float
    status: RecognitionStatus


class Worker(threading.Thread):

    def __init__(self, job_q: Queue, result_q: Queue):
        super().__init__(daemon=True)
        self.job_q = job_q
        self.result_q = result_q
        self.ocr = easyocr.Reader(['en'])

    def run(self):
        while True:
            job = self.job_q.get()

            if job is None:
                break

            try:
                text = "DEMO123"  # placeholder OCR
                self.result_q.put(
                    RecognitionResult(
                        job.job_id,
                        job.track_id,
                        text,
                        0.9,
                        RecognitionStatus.SUCCESS
                    )
                )
            except:
                pass


class WorkerPoolStage:

    def __init__(self, job_q, result_q, num_workers=2):
        self.workers = [
            Worker(job_q, result_q) for _ in range(num_workers)
        ]

    def start(self):
        for w in self.workers:
            w.start()

    def shutdown(self):
        for w in self.workers:
            w.job_q.put(None)