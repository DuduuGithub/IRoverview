"""
API REQUEST PARALLEL PROCESSOR

This script parallelizes requests to the OpenAI API while throttling to stay under rate limits.

Features:
- Streams requests from file, to avoid running out of memory for giant jobs
- Makes requests concurrently, to maximize throughput
- Throttles request and token usage, to stay under rate limits
- Retries failed requests up to {max_attempts} times, to avoid missing data
- Logs errors, to diagnose problems with requests

Inputs:
- requests_filepath : str
    - path to the file containing the requests to be processed
    - file should be a jsonl file, where each line is a json object with API parameters and an optional metadata field
    - e.g., {"model": "text-embedding-3-small", "input": "embed me", "metadata": {"row_id": 1}}
    - as with all jsonl files, take care that newlines in the content are properly escaped (json.dumps does this automatically)
    - an example file is provided at examples/data/example_requests_to_parallel_process.jsonl
    - the code to generate the example file is appended to the bottom of this script
- save_filepath : str, optional
    - path to the file where the results will be saved
    - file will be a jsonl file, where each line is an array with the original request plus the API response
    - e.g., [{"model": "text-embedding-3-small", "input": "embed me"}, {...}]
    - if omitted, results will be saved to {requests_filename}_results.jsonl
- request_url : str, optional
    - URL of the API endpoint to call
    - if omitted, will default to "https://api.openai.com/v1/embeddings"
- api_key : str, optional
    - API key to use
    - if omitted, the script will attempt to read it from an environment variable {os.getenv("OPENAI_API_KEY")}
- max_requests_per_minute : float, optional
    - target number of requests to make per minute (will make less if limited by tokens)
    - leave headroom by setting this to 50% or 75% of your limit
    - if requests are limiting you, try batching multiple embeddings or completions into one request
    - if omitted, will default to 1,500
- max_tokens_per_minute : float, optional
    - target number of tokens to use per minute (will use less if limited by requests)
    - leave headroom by setting this to 50% or 75% of your limit
    - if omitted, will default to 125,000
- token_encoding_name : str, optional
    - name of the token encoding used, as defined in the `tiktoken` package
    - if omitted, will default to "cl100k_base" (used by `text-embedding-3-small`)
- max_attempts : int, optional
    - number of times to retry a failed request before giving up
    - if omitted, will default to 5
- logging_level : int, optional
    - level of logging to use; higher numbers will log fewer messages
    - 40 = ERROR; will log only when requests fail after all retries
    - 30 = WARNING; will log when requests his rate limits or other errors
    - 20 = INFO; will log when requests start and the status at finish
    - 10 = DEBUG; will log various things as the loop runs to see when they occur
    - if omitted, will default to 20 (INFO).

The script is structured as follows:
    - Imports
    - Define main()
        - Initialize things
        - In main loop:
            - Get next request if one is not already waiting for capacity
            - Update available token & request capacity
            - If enough capacity available, call API
            - The loop pauses if a rate limit error is hit
            - The loop breaks when no tasks remain
    - Define dataclasses
        - StatusTracker (stores script metadata counters; only one instance is created)
        - APIRequest (stores API inputs, outputs, metadata; one method to call API)
    - Define functions
        - api_endpoint_from_url (extracts API endpoint from request URL)
        - append_to_jsonl (writes to results file)
        - num_tokens_consumed_from_request (bigger function to infer token usage from request)
        - task_id_generator_function (yields 0, 1, 2, ...)
    - Run main()
"""

# imports
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata
from typing import Dict, List, Optional, Callable


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: int = 3_500,
    max_tokens_per_minute: int = 90_000,
    token_encoding_name: str = "cl100k_base",
    max_attempts: int = 5,
    logging_level: int = logging.INFO
):
    """
    Processes API requests from a JSONL file at the specified rate and saves results to another JSONL file.
    
    Args:
        requests_filepath: Path to the JSONL file containing the requests.
        save_filepath: Path where the JSONL file with results will be saved.
        request_url: URL for the API endpoint.
        api_key: API key for the endpoint.
        max_requests_per_minute: Maximum number of requests per minute.
        max_tokens_per_minute: Maximum number of tokens per minute.
        token_encoding_name: Name of the encoding used for counting tokens.
        max_attempts: Maximum number of attempts for each request.
        logging_level: Logging level.
    """
    # Set up logging
    logging.basicConfig(level=logging_level)
    logging.info(f"Processing requests from {requests_filepath} and saving to {save_filepath}")
    
    # Load requests from jsonl file
    with open(requests_filepath, 'r') as f:
        requests_data = [json.loads(line) for line in f if line.strip()]
    
    # Create save file if it doesn't exist
    if not os.path.exists(save_filepath):
        with open(save_filepath, 'w') as f:
            pass
    
    # Load already processed requests
    processed_requests = set()
    if os.path.exists(save_filepath):
        with open(save_filepath, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if isinstance(data, list) and len(data) > 2 and 'original_index' in data[2]:
                            processed_requests.add(data[2]['original_index'])
                    except json.JSONDecodeError:
                        pass

    # Filter out already processed requests
    requests_to_process = []
    for i, request_data in enumerate(requests_data):
        if 'metadata' in request_data and 'original_index' in request_data['metadata']:
            if request_data['metadata']['original_index'] not in processed_requests:
                requests_to_process.append((i, request_data))
        else:
            requests_to_process.append((i, request_data))
    
    if not requests_to_process:
        logging.info("No new requests to process. Exiting.")
        return
    
    logging.info(f"Processing {len(requests_to_process)} new requests out of {len(requests_data)} total.")
    
    # Set up rate limiting
    request_interval = 60.0 / max_requests_per_minute
    token_interval = 60.0 / max_tokens_per_minute
    
    # Process requests
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, request_data in requests_to_process:
            task = asyncio.create_task(
                process_api_request(
                    request_data=request_data,
                    request_index=i,
                    save_filepath=save_filepath,
                    request_url=request_url,
                    api_key=api_key,
                    request_interval=request_interval,
                    token_interval=token_interval,
                    max_attempts=max_attempts,
                    session=session
                )
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    logging.info(f"Completed processing requests from {requests_filepath}")

async def process_api_request(
    request_data: Dict,
    request_index: int,
    save_filepath: str,
    request_url: str,
    api_key: str,
    request_interval: float,
    token_interval: float,
    max_attempts: int,
    session: aiohttp.ClientSession
):
    """
    Process a single API request with retries and rate limiting
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    for attempt in range(max_attempts):
        try:
            # Add delay based on rate limiting
            await asyncio.sleep(request_interval)
            
            # Make the API request
            async with session.post(request_url, headers=headers, json=request_data) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Save result to file
                    with open(save_filepath, 'a') as f:
                        # Save the request and result pair
                        f.write(json.dumps([request_data, result, request_data.get('metadata', {})]) + '\n')
                    
                    # Successful request, break the retry loop
                    break
                elif response.status == 429:
                    # Rate limit hit - wait longer
                    wait_time = 2 ** attempt * 10  # Exponential backoff
                    logging.warning(f"Rate limit hit, waiting {wait_time} seconds")
                    await asyncio.sleep(wait_time)
                else:
                    error_detail = await response.text()
                    logging.error(f"Error {response.status}: {error_detail}")
                    wait_time = 2 ** attempt * 5  # Exponential backoff
                    await asyncio.sleep(wait_time)
        except Exception as e:
            logging.error(f"Request error: {str(e)}")
            if attempt + 1 < max_attempts:
                wait_time = 2 ** attempt * 5
                await asyncio.sleep(wait_time)
            else:
                logging.error(f"Failed after {max_attempts} attempts: {str(e)}")
                # Save error record to file
                with open(save_filepath, 'a') as f:
                    error_result = {"error": str(e)}
                    f.write(json.dumps([request_data, error_result, request_data.get('metadata', {})]) + '\n')


# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        # logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "rate limit" in response["error"].get("message", "").lower():
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


# functions


def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(
            r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url
        )
    return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request'
                )
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1