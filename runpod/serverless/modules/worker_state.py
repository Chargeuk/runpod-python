"""
Handles getting stuff from environment variables and updating the global state like job id.
"""

import os
import time
import uuid
from typing import Any, Dict, Optional

from .rp_logger import RunPodLogger
import threading
from collections import deque
import threading
import copy


log = RunPodLogger()

REF_COUNT_ZERO = time.perf_counter()  # Used for benchmarking with the debugger.

WORKER_ID = os.environ.get("RUNPOD_POD_ID", str(uuid.uuid4()))


# ----------------------------------- Flags ---------------------------------- #
IS_LOCAL_TEST = os.environ.get("RUNPOD_WEBHOOK_GET_JOB", None) is None


# ------------------------------- Job Tracking ------------------------------- #



class Job:
    """
    Represents a job object.

    Args:
        id: The id of the job, a unique string.
        input: The input to the job.
        webhook: The webhook to send the job output to.
        status: The status of the job (default: "QUEUED").
        created_time: The time the job was created (epoch seconds).
        status_modified_time: The last time the status was changed (epoch seconds).
        started_time: The time the job status was set to IN_PROGRESS (epoch seconds).
        completed_time: The time the job status was set to COMPLETED or FAILED (epoch seconds).
    """

    def __init__(
        self,
        id: str,
        input: Optional[Dict[str, Any]] = None,
        webhook: Optional[str] = None,
        status: str = "QUEUED",
        **kwargs
    ) -> None:
        now = time.time()
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "input", input)
        object.__setattr__(self, "webhook", webhook)
        object.__setattr__(self, "created_time", now)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "status_modified_time", now)
        object.__setattr__(self, "started_time", None)
        object.__setattr__(self, "completed_time", None)
        object.__setattr__(self, "error", None)  # <-- Add this line
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)

    def __getattribute__(self, name):
        _lock = object.__getattribute__(self, "_lock")
        with _lock:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        _lock = object.__getattribute__(self, "_lock")
        with _lock:
            if name == "status":
                object.__setattr__(self, "status_modified_time", time.time())
                # Set started_time if status is set to IN_PROGRESS and not already set
                if value == "IN_PROGRESS" and getattr(self, "started_time", None) is None:
                    object.__setattr__(self, "started_time", time.time())
                # Set completed_time if status is set to COMPLETED or FAILED and not already set
                if value in ("COMPLETED", "FAILED") and getattr(self, "completed_time", None) is None:
                    object.__setattr__(self, "completed_time", time.time())
            object.__setattr__(self, name, value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Job):
            return self.id == other.id
        return False

    def __hash__(self) -> int:
        return hash(self.id)

    def __str__(self) -> str:
        return self.id

    def getString(self) -> str:
        """
        Returns a string representation of all fields that are not None, thread-safe,
        excluding the 'input' field.
        """
        with self._lock:
            fields = []
            for key in self.__dict__:
                if key in ("_lock", "input"):
                    continue
                value = getattr(self, key)
                if value is not None:
                    fields.append(f"{key}={value!r}")
            return f"Job({', '.join(fields)})"

    def getDictCopy(self, includeNone=True) -> dict:
        """
        Returns a deep copy of the job's __dict__ (excluding the lock), thread-safe.
        Only includes fields with a value of None if includeNone is True.
        """
        with self._lock:
            if includeNone:
                d = {k: v for k, v in self.__dict__.items() if k != "_lock"}
            else:
                d = {k: v for k, v in self.__dict__.items() if k != "_lock" and v is not None}
            return copy.deepcopy(d)

# ---------------------------------------------------------------------------- #
#                                    Tracker                                   #
# ---------------------------------------------------------------------------- #



class JobsProgress:
    """Track the state of current jobs in progress, thread-safe and ordered."""

    _instance = None

    def __new__(cls):
        if JobsProgress._instance is None:
            JobsProgress._instance = super().__new__(cls)
            JobsProgress._instance._init_internal()
        return JobsProgress._instance

    def _init_internal(self):
        self._jobs = deque()  # ordered queue of Job objects
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>: {self.get_job_list()}"

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()

    def add(self, element: Any):
        """
        Adds a Job object to the queue.

        If the added element is a string, then `Job(id=element)` is added
        
        If the added element is a dict, then `Job(**element)` is added
        """
        if isinstance(element, str):
            element = Job(id=element)
        elif isinstance(element, dict):
            element = Job(**element)

        if not isinstance(element, Job):
            raise TypeError("Only Job objects can be added to JobsProgress.")

        with self._lock:
            # Prevent duplicates by job id
            if not any(job.id == element.id for job in self._jobs):
                self._jobs.append(element)

    def remove(self, element: Any):
        """
        Removes a Job object from the queue.

        If the element is a string, then `Job(id=element)` is removed
        
        If the element is a dict, then `Job(**element)` is removed
        """
        if isinstance(element, str):
            job_id = element
        elif isinstance(element, dict):
            job_id = element.get("id")
        elif isinstance(element, Job):
            job_id = element.id
        else:
            raise TypeError("Only Job objects can be removed from JobsProgress.")

        with self._lock:
            self._jobs = deque(job for job in self._jobs if job.id != job_id)

    def get(self, element: Any) -> Optional[Job]:
        if isinstance(element, str):
            job_id = element
        elif isinstance(element, dict):
            job_id = element.get("id")
        elif isinstance(element, Job):
            job_id = element.id
        else:
            raise TypeError("Only Job objects can be retrieved from JobsProgress.")

        with self._lock:
            for job in self._jobs:
                if job.id == job_id:
                    return job
        return None

    def get_job_list(self) -> str:
        """
        Returns the list of job IDs as comma-separated string.
        """
        with self._lock:
            if not self._jobs:
                return None
            return ",".join(str(job) for job in self._jobs)

    def get_job_count(self) -> int:
        """
        Returns the number of jobs.
        """
        with self._lock:
            return len(self._jobs)
        
    def update_status(self, job_id: str, status: str):
        with self._lock:
            for job in self._jobs:
                if job.id == job_id:
                    job.status = status
                    break

    def get_status(self, job_id: str) -> Optional[str]:
        with self._lock:
            for job in self._jobs:
                if job.id == job_id:
                    return getattr(job, "status", None)
        return None

    def pop_oldest(self) -> Optional[Job]:
        """
        Removes and returns the oldest job in the queue, or None if empty.
        """
        with self._lock:
            if self._jobs:
                return self._jobs.popleft()
            return None
        
    def get_oldest_queued(self) -> Optional["Job"]:
        """
        Returns the oldest job in QUEUED state without removing it.
        """
        with self._lock:
            for job in self._jobs:
                if getattr(job, "status", None) == "QUEUED":
                    return job
        return None
