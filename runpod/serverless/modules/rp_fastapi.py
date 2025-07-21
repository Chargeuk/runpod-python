""" Used to launch the FastAPI web server when worker is running in API mode. """

import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import signal
import traceback
import subprocess

import requests
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse

from ...http_client import SyncClientSession
from ...version import __version__ as runpod_version
from .rp_handler import is_generator
from .rp_job import run_job, run_job_generator
from .rp_ping import Heartbeat
from .worker_state import Job, JobsProgress
from .rp_logger import RunPodLogger
import asyncio

RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", None)

TITLE = "RunPod | Development Worker API"

DESCRIPTION = """
The Development Worker API facilitates testing and debugging of your RunPod workers.
It offers a sandbox environment for executing code and simulating interactions with your worker, ensuring your applications can seamlessly transition to production on RunPod serverless platform.
Use this API for comprehensive testing of request submissions and result retrieval, mimicking the behavior of RunPod's operational environment.
---
*Note: This API serves as a local testing tool and will not be utilized once your worker is operational on the RunPod platform.*
"""

# Add CLI tool suggestion if RUNPOD_PROJECT_ID is not set.
if os.environ.get("RUNPOD_PROJECT_ID", None) is None:
    DESCRIPTION += """

    ℹ️ | Consider developing with our CLI tool to streamline your worker development process.

    >_  wget -qO- cli.runpod.net | sudo bash
    >_  runpodctl project create
    """

RUN_DESCRIPTION = """
Initiates processing jobs, returning a unique job ID.

**Parameters:**
- **input** (string): The data to be processed by the worker. This could be a string, JSON object, etc., depending on the worker's requirements.
- **webhook** (string, optional): A callback URL for result notification upon completion. If specified, the server will send a POST request to this URL with the job's result once it's available.

**Returns:**
- **job_id** (string): A unique identifier for the job, used with the `/stream` and `/status` endpoints for monitoring progress and checking job status.
"""

RUNSYNC_DESCRIPTION = """
Executes processing jobs synchronously, returning the job's output directly.

This endpoint is ideal for tasks where immediate result retrieval is necessary,
streamlining the execution process by eliminating the need for subsequent
status or result checks.

**Parameters:**
- **input** (string): The data to be processed by the worker. This should be in a format that the worker can understand (e.g., JSON, text, etc.).
- **webhook** (string, optional): A callback URL to which the result will be posted. While direct result retrieval is the primary operation mode for this endpoint, specifying a webhook allows for asynchronous result notification if needed.

**Returns:**
- **output** (Any): The direct output from the processing job, formatted according to the job's nature and the expected response structure. This could be a JSON object, plain text, or any data structure depending on the processing logic.
"""

STREAM_DESCRIPTION = """
Continuously aggregates the output of a processing job, returning the full output once the job is complete.

This endpoint is especially useful for jobs where the complete output needs to be accessed at once. It provides a consolidated view of the results post-completion, ensuring that users can retrieve the entire output without the need to poll multiple times or manage partial results.

**Parameters:**
- **job_id** (string): The unique identifier of the job for which output is being requested. This ID is used to track the job's progress and aggregate its output.

**Returns:**
- **output** (Any): The aggregated output from the job, returned as a single entity once the job has concluded. The format of the output will depend on the nature of the job and how its results are structured.
"""

STATUS_DESCRIPTION = """
Checks the completion status of a processing job and returns its output if the job is complete.

This endpoint is invaluable for monitoring the progress of a job and obtaining the output only after the job has fully completed. It simplifies the process of querying job completion and retrieving results, eliminating the need for continuous polling or result aggregation.

**Parameters:**
- **job_id** (string): The unique identifier for the job being queried. This ID is used to track and assess the status of the job.

**Returns:**
- **status** (string): The completion status of the job, typically 'complete' or 'in progress'. This status indicates whether the job has finished processing and if the output is ready for retrieval.
- **output** (Any, optional): The final output of the job, provided if the job is complete. The format and structure of the output depend on the job's nature and the data processing involved.

**Note:** The availability of the `output` field is contingent on the job's completion status. If the job is still in progress, this field may be omitted or contain partial results, depending on the implementation.
"""

API_STATUS_DESCRIPTION = """
Returns the current status of the API system, including information about the worker's CUDA support, job queue, and system health.

**Parameters:**
- **vram_threshold_gb** (float, optional): Override the VRAM usage threshold in GB. If not provided, uses the RUNPOD_VRAM_THRESHOLD_GB environment variable (default: 12.0).

**Returns:**
- **cuda_available** (boolean): Is cuda available on the worker?
- **cuda_functional** (boolean): Can CUDA actually perform computations successfully?
- **vram_usage_gb** (float): Current VRAM usage in gigabytes.
- **vram_total_gb** (float): Total VRAM available in gigabytes.
- **vram_usage_percent** (float): Current VRAM usage as a percentage of total VRAM.
- **vram_check_passed** (boolean): Whether VRAM usage is below the configured threshold.
- **cuda_version** (string): The version of CUDA available on the worker.
- **job_queue_length** (int): The number of jobs currently in the queue.
- **system_health** (string): The health status of the system, indicating whether it is operational or experiencing issues.
- **pytorch_version** (string): The version of PyTorch installed on the worker.
- **pytorch_audio_version** (string): The version of PyTorch Audio installed on the worker.
- **pytorch_vision_version** (string): The version of PyTorch Vision installed on the worker.
- **nvidia_smi_output** (string): The output of the `nvidia-smi --version` command, providing details about the GPU status.
- **nvidia_smi_gpu** (string): The GPU model and status as reported by `nvidia-smi -L`.
- **ollama_status** (string): The status of the Ollama service, indicating whether it is running or not.
- **comfyui_status** (string): The status of the ComfyUI service, indicating whether it is running or not.

**Environment Variables:**
- **RUNPOD_VRAM_THRESHOLD_GB**: Sets the VRAM usage threshold in GB (default: 12.0). If VRAM usage exceeds this threshold, the vram_check_passed will be false and system_health will be "Degraded".
"""


# ------------------------------ Initializations ----------------------------- #
job_list = JobsProgress()
heartbeat = Heartbeat()
log = RunPodLogger()


# ------------------------------- Input Objects ------------------------------ #
# @dataclass
# class Job:
#     """Represents a job."""

#     id: str
#     input: Union[dict, list, str, int, float, bool]
#     status: str = "QUEUED"  # Add status field


# @dataclass
# class TestJob:
#     """Represents a test job.
#     input can be any type of data.
#     """

#     id: Optional[str] = None
#     input: Optional[Union[dict, list, str, int, float, bool]] = None
#     webhook: Optional[str] = None
#     status: str = "QUEUED"  # Add status field


@dataclass
class DefaultRequest:
    """Represents a test input."""

    input: Dict[str, Any]
    webhook: Optional[str] = None


@dataclass
class ApiStatusRequest:
    """Represents an API status request."""

    vram_threshold_gb: Optional[float] = None


# ------------------------------ Output Objects ------------------------------ #
@dataclass
class JobOutput:
    """Represents the output of a job."""

    id: str
    status: str
    output: Optional[Union[dict, list, str, int, float, bool]] = None
    error: Optional[str] = None


@dataclass
class StreamOutput:
    """Stream representation of a job."""

    id: str
    status: str = "IN_PROGRESS"
    stream: Optional[Union[dict, list, str, int, float, bool]] = None
    error: Optional[str] = None


# ------------------------------ Webhook Sender ------------------------------ #
def _send_webhook(url: str, payload: Dict[str, Any]) -> bool:
    """
    Sends a webhook to the provided URL.

    Args:
        url (str): The URL to send the webhook to.
        payload (Dict[str, Any]): The JSON payload to send.

    Returns:
        bool: True if the request was successful, False otherwise.
    """
    with SyncClientSession() as session:
        try:
            response = session.post(url, json=payload, timeout=10)
            response.raise_for_status()  # Raises exception for 4xx/5xx responses
            return True
        except requests.RequestException as err:
            print(f"WEBHOOK | Request to {url} failed: {err}")
            return False


# ---------------------------------------------------------------------------- #
#                                  API Worker                                  #
# ---------------------------------------------------------------------------- #
class WorkerAPI:
    """Used to launch the FastAPI web server when the worker is running in API mode."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the WorkerAPI class.
        1. Starts the heartbeat thread.
        2. Initializes the FastAPI web server.
        3. Sets the handler for processing jobs.
        """
        # Start the heartbeat thread.
        heartbeat.start_ping()

        self.config = config

        tags_metadata = [
            {
                "name": "Synchronously Submit Request & Get Job Results",
                "description": "Endpoints for submitting job requests and getting the results.",
            },
            {
                "name": "Submit Job Requests",
                "description": "Endpoints for submitting job requests.",
            },
            {
                "name": "Check Job Results",
                "description": "Endpoints for checking the status of a job and getting the results.",
            },
        ]

        # Initialize the FastAPI web server.
        self.rp_app = FastAPI(
            title=TITLE,
            description=DESCRIPTION,
            version=runpod_version,
            docs_url="/",
            openapi_tags=tags_metadata,
        )

        # Create an APIRouter and add the route for processing jobs.
        api_router = APIRouter()

        # Docs Redirect /docs -> /
        api_router.add_api_route(
            "/docs", lambda: RedirectResponse(url="/"), include_in_schema=False
        )

        if RUNPOD_ENDPOINT_ID:
            api_router.add_api_route(
                f"/{RUNPOD_ENDPOINT_ID}/realtime", self._realtime, methods=["POST"]
            )

        # Simulation endpoints.
        api_router.add_api_route(
            "/run",
            self._sim_run,
            methods=["POST"],
            # response_model_exclude_none=True,
            summary="Mimics the behavior of the run endpoint.",
            description=RUN_DESCRIPTION,
            tags=["Submit Job Requests"],
        )
        api_router.add_api_route(
            "/runsync",
            self._sim_runsync,
            methods=["POST"],
            # response_model_exclude_none=True,
            summary="Mimics the behavior of the runsync endpoint.",
            description=RUNSYNC_DESCRIPTION,
            tags=["Synchronously Submit Request & Get Job Results"],
        )
        api_router.add_api_route(
            "/stream/{job_id}",
            self._sim_stream,
            methods=["POST"],
            # response_model_exclude_none=True,
            summary="Mimics the behavior of the stream endpoint.",
            description=STREAM_DESCRIPTION,
            tags=["Check Job Results"],
        )
        api_router.add_api_route(
            "/status/{job_id}",
            self._sim_status,
            methods=["POST"],
            # response_model_exclude_none=True,
            summary="Mimics the behavior of the status endpoint.",
            description=STATUS_DESCRIPTION,
            tags=["Check Job Results"],
        )
        api_router.add_api_route(
            "/apistatus",
            self._sim_api_status,
            methods=["POST"],
            # response_model_exclude_none=True,
            summary="Returns information about the api system status.",
            description=API_STATUS_DESCRIPTION,
            tags=["Api Information"],
        )

        # Include the APIRouter in the FastAPI application.
        self.rp_app.include_router(api_router)

         # Start background job runner thread
        self._stop_event = threading.Event()
        self._job_runner_thread = threading.Thread(
            target=self._background_job_runner,
            daemon=True
        )
        self._job_runner_thread.start()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)

    def _background_job_runner(self):
        """Continuously processes jobs from the queue."""
        while not self._stop_event.is_set():
            job = job_list.get_oldest_queued()
            if job is None:
                time.sleep(0.1)
                continue

            log.info(f"WorkerApi1 procesing job {job.getString()}")
            job.status = "IN_PROGRESS"

            try:
                log.info(f"WorkerApi procesing job {job.getString()}", job.id)
                if is_generator(self.config["handler"]):
                    log.info(f"WorkerApi procesing job as generator", job.id)

                    def run_async_generator(gen):
                        results = []
                        async def consume():
                            async for stream_output in gen:
                                results.append(stream_output["output"])
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(consume())
                        loop.close()
                        return results

                    generator_output = run_job_generator(self.config["handler"], job.getDictCopy())
                    job_output = {"output": run_async_generator(generator_output)}
                else:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    log.info(f"WorkerApi procesing job as NON-generator", job.id)
                    job_output = loop.run_until_complete(run_job(self.config["handler"], job.getDictCopy()))
                    loop.close()

                if job_output.get("error", None):
                    job.status = "FAILED"
                    job.error = job_output.get("error", "Unknown error")
                    log.info(f"WorkerApi Job failed {job.getString()}", job.id)
                else:
                    job.status = "COMPLETED"
                    log.info(f"WorkerApi Job completed {job.getString()}", job.id)

                if job.webhook:
                    thread = threading.Thread(
                        target=_send_webhook,
                        args=(job.webhook, job_output),
                        daemon=True,
                    )
                    thread.start()
            except Exception as e:
                stack = traceback.format_exc()
                log.error(f"WorkerApi Job failed {job.getString()}. error: {e}\nStacktrace:\n{stack}")
                job.status = "FAILED"
                job.error = str(e)

    def stop(self):
        """Stops the background job runner thread."""
        self._stop_event.set()
        self._job_runner_thread.join()

    def start_uvicorn(self, api_host="localhost", api_port=8000, api_concurrency=1):
        """
        Starts the Uvicorn server.
        """
        uvicorn.run(
            self.rp_app,
            host=api_host,
            port=int(api_port),
            workers=int(api_concurrency),
            log_level=os.environ.get("UVICORN_LOG_LEVEL", "info"),
            access_log=False,
        )

    # ----------------------------- Realtime Endpoint ---------------------------- #
    async def _realtime(self, job: Job):
        """
        Performs model inference on the input data using the provided handler.
        If handler is not provided, returns an error message.
        """
        job_list.add(job.id)

        # Process the job using the provided handler, passing in the job input.
        job_results = await run_job(self.config["handler"], job.getDictCopy())

        job_list.remove(job.id)

        # Return the results of the job processing.
        return jsonable_encoder(job_results)

    # ---------------------------------------------------------------------------- #
    #                             Simulation Endpoints                             #
    # ---------------------------------------------------------------------------- #

    # ------------------------------------ run ----------------------------------- #
    async def _sim_run(self, job_request: DefaultRequest):
        assigned_job_id = f"test-{uuid.uuid4()}"
        job = Job(id=assigned_job_id, input=job_request.input, webhook=job_request.webhook, status="QUEUED")
        job_list.add(job)
        response = self._build_status_response(job)
        return jsonable_encoder(response)

    # ---------------------------------- runsync --------------------------------- #
    async def _sim_runsync(self, job_request: DefaultRequest):
        """Development endpoint to simulate runsync behavior."""
        assigned_job_id = f"test-{uuid.uuid4()}"
        job = Job(id=assigned_job_id, input=job_request.input)

        if is_generator(self.config["handler"]):
            generator_output = run_job_generator(self.config["handler"], job.getDictCopy())
            job_output = {"output": []}
            async for stream_output in generator_output:
                job_output["output"].append(stream_output["output"])
        else:
            job_output = await run_job(self.config["handler"], job.getDictCopy())

        if job_output.get("error", None):
            return jsonable_encoder(
                {"id": job.id, "status": "FAILED", "error": job_output["error"]}
            )

        if job_request.webhook:
            thread = threading.Thread(
                target=_send_webhook,
                args=(job_request.webhook, job_output),
                daemon=True,
            )
            thread.start()

        return jsonable_encoder(
            {"id": job.id, "status": "COMPLETED", "output": job_output["output"]}
        )

    # ---------------------------------- stream ---------------------------------- #
    async def _sim_stream(self, job_id: str) -> StreamOutput:
        """Development endpoint to simulate stream behavior."""
        stashed_job = job_list.get(job_id)
        if stashed_job is None:
            return jsonable_encoder(
                {"id": job_id, "status": "FAILED", "error": "Job ID not found"}
            )

        job = Job(id=job_id, input=stashed_job.input)

        if is_generator(self.config["handler"]):
            generator_output = run_job_generator(self.config["handler"], job.getDictCopy())
            stream_accumulator = []
            async for stream_output in generator_output:
                stream_accumulator.append({"output": stream_output["output"]})
        else:
            return jsonable_encoder(
                {
                    "id": job_id,
                    "status": "FAILED",
                    "error": "Stream not supported, handler must be a generator.",
                }
            )

        job_list.remove(job.id)

        if stashed_job.webhook:
            thread = threading.Thread(
                target=_send_webhook,
                args=(stashed_job.webhook, stream_accumulator),
                daemon=True,
            )
            thread.start()

        return jsonable_encoder(
            {"id": job_id, "status": "COMPLETED", "stream": stream_accumulator}
        )

    # ---------------------------------- status ---------------------------------- #
    def _build_status_response(self, job) -> dict:
        """Builds a status response dictionary from a Job object."""
        response = job.getDictCopy(includeNone=False)

        # Remove 'input' from the response if present
        response.pop("input", None)

        # Calculate delayTime if possible
        started_time = response.get("started_time")
        created_time = response.get("created_time")
        current_time = time.time()
        response["currentTime"] = current_time
        if started_time is not None and created_time is not None:
            response["delayTime"] = started_time - created_time

        if started_time is not None and current_time is not None:
            response["timeSoFar"] = current_time - started_time

        # Calculate executionTime if possible
        completed_time = response.get("completed_time")
        if completed_time is not None and started_time is not None:
            response["executionTime"] = completed_time - started_time

        log.debug(f"Status response: {response}", job.id)
        return response

    async def _sim_status(self, job_id: str):
        """Development endpoint to simulate status behavior."""
        stashed_job = job_list.get(job_id)
        if stashed_job is None:
            return jsonable_encoder(
                {"id": job_id, "status": "FAILED", "error": "Job ID not found"}
            )

        response = self._build_status_response(stashed_job)
        returnedResponse = jsonable_encoder(response)
        log.debug(f"Returned Status response: {returnedResponse}", job_id)
        return returnedResponse
    

    # --------------------------------- apistatus -------------------------------- #
    async def _sim_api_status(self, request: Optional[ApiStatusRequest] = None):
        """
        Returns the current status of the API system, including information about the worker's CUDA support, job queue, and system health.
        """
        # Extract vram_threshold_gb from request if provided
        vram_threshold_gb = request.vram_threshold_gb if request else None
        # Set initial defaults
        cuda_available = False
        cuda_functional = False
        vram_usage_gb = 0.0
        vram_total_gb = 0.0
        vram_usage_percent = 0.0
        vram_check_passed = True
        cuda_version = "N/A"
        pytorch_version = "N/A"
        pytorch_audio_version = "N/A"
        pytorch_vision_version = "N/A"
        nvidia_smi_output = "N/A"
        nvidia_smi_gpu = "N/A"
        ollama_status = "N/A"
        comfyui_status = "N/A"
        job_queue_length = 0
        isRunning = True
        
        # VRAM usage threshold in GB (configurable via environment variable)
        if vram_threshold_gb is None:
            vram_threshold_gb = float(os.environ.get("RUNPOD_VRAM_THRESHOLD_GB", "3.0"))
        # Try to get torch and related versions and CUDA info
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else "N/A"
            pytorch_version = torch.__version__
            
            # Test actual CUDA functionality
            if cuda_available:
                try:
                    # Create a small tensor on GPU and perform a simple operation
                    test_tensor = torch.randn(10, 10, device='cuda')
                    result = torch.matmul(test_tensor, test_tensor.T)
                    # Synchronize to ensure operation completed successfully
                    torch.cuda.synchronize()
                    cuda_functional = True
                    
                    # Check VRAM usage
                    try:
                        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        vram_allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
                        vram_reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
                        vram_usage_gb = max(vram_allocated_gb, vram_reserved_gb)
                        vram_usage_percent = (vram_usage_gb / vram_total_gb) * 100 if vram_total_gb > 0 else 0
                        
                        if vram_usage_gb > vram_threshold_gb:
                            vram_check_passed = False
                            log.warning(f"VRAM usage check failed: {vram_usage_gb:.2f}GB used > {vram_threshold_gb}GB threshold")
                            isRunning = False
                        else:
                            log.info(f"VRAM usage check passed: {vram_usage_gb:.2f}GB used <= {vram_threshold_gb}GB threshold")
                            
                    except Exception as vram_error:
                        log.warning(f"VRAM usage check failed: {vram_error}")
                        vram_check_passed = False
                        isRunning = False
                        
                except Exception as cuda_test_error:
                    log.warning(f"CUDA functional test failed: {cuda_test_error}")
                    cuda_functional = False
                    isRunning = False
        except Exception as e:
            log.warning(f"Could not get torch or CUDA info: {e}")
            isRunning = False

        try:
            import torchaudio as torch_audio
            pytorch_audio_version = torch_audio.__version__
        except Exception as e:
            log.info(f"torchaudio not available: {e}")
            isRunning = False

        try:
            import torchvision as torch_vision
            pytorch_vision_version = torch_vision.__version__
        except Exception as e:
            log.info(f"torchvision not available: {e}")
            isRunning = False

        try:
            result = subprocess.run(
                ["nvidia-smi", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            nvidia_smi_output = result.stdout
            log.info(nvidia_smi_output)
        except subprocess.CalledProcessError as e:
            log.info("nvidia-smi failed:", e.stderr)
            isRunning = False
        except FileNotFoundError:
            log.info("nvidia-smi command not found (NVIDIA drivers may not be installed)")
            isRunning = False

        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            nvidia_smi_gpu = result.stdout
            log.info(nvidia_smi_gpu)
        except subprocess.CalledProcessError as e:
            log.info("nvidia-smi failed:", e.stderr)
            isRunning = False
        except FileNotFoundError:
            log.info("nvidia-smi command not found (NVIDIA drivers may not be installed)")
            isRunning = False

        try:
            job_queue_length = job_list.get_job_count()
        except Exception as e:
            log.info(f"Unable to get job queue length: {e}")
            isRunning = False

        # Check if Ollama service is running
        try:
            result = subprocess.run(
                ["pgrep", "-f", "ollama serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False  # Don't raise exception on non-zero exit
            )
            if result.returncode == 0:
                ollama_status = "Running"
            else:
                ollama_status = "Not Running"
        except FileNotFoundError:
            # pgrep command not found (likely on Windows)
            try:
                # Alternative check for Windows using tasklist
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq ollama.exe"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                if "ollama.exe" in result.stdout:
                    ollama_status = "Running"
                else:
                    ollama_status = "Not Running"
            except Exception as e:
                log.info(f"Unable to check Ollama status: {e}")
                ollama_status = "Unknown"
        except Exception as e:
            log.info(f"Unable to check Ollama status: {e}")
            ollama_status = "Unknown"

        # Check if ComfyUI service is running
        try:
            result = subprocess.run(
                ["pgrep", "-f", "python.*main.py.*--port.*3000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False  # Don't raise exception on non-zero exit
            )
            if result.returncode == 0:
                comfyui_status = "Running"
            else:
                comfyui_status = "Not Running"
        except FileNotFoundError:
            # pgrep command not found (likely on Windows)
            try:
                # Alternative check for Windows using tasklist and findstr
                result = subprocess.run(
                    ["tasklist", "/V"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                # Check if any python process has main.py and port 3000 in command line
                if "python" in result.stdout and "main.py" in result.stdout and "3000" in result.stdout:
                    comfyui_status = "Running"
                else:
                    comfyui_status = "Not Running"
            except Exception as e:
                log.info(f"Unable to check ComfyUI status: {e}")
                comfyui_status = "Unknown"
        except Exception as e:
            log.info(f"Unable to check ComfyUI status: {e}")
            comfyui_status = "Unknown"

        if ollama_status != "Running":
            isRunning = False
        if comfyui_status != "Running":
            isRunning = False

        system_health = "Operational" if isRunning else "Degraded"
        response = {
            "cuda_available": cuda_available,
            "cuda_functional": cuda_functional,
            "vram_usage_gb": vram_usage_gb,
            "vram_total_gb": vram_total_gb,
            "vram_usage_percent": vram_usage_percent,
            "vram_check_passed": vram_check_passed,
            "cuda_version": cuda_version,
            "job_queue_length": job_queue_length,
            "system_health": system_health,
            "pytorch_version": pytorch_version,
            "pytorch_audio_version": pytorch_audio_version,
            "pytorch_vision_version": pytorch_vision_version,
            "nvidia_smi_output": nvidia_smi_output,
            "nvidia_smi_gpu": nvidia_smi_gpu,
            "ollama_status": ollama_status,
            "comfyui_status": comfyui_status,
        }
        log.info(f"API Status response: {response}")
        return jsonable_encoder(response)
