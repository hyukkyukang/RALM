import logging
import queue
import threading

import numpy as np

logger = logging.getLogger("AsyncSaver")


class AsyncEmbeddingSaver:
    """
    Saves data to disk asynchronously using a background thread.
    """

    def __init__(self, max_queue_size: int = 3):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True  # Allow thread to exit if main process exits
        self.thread.start()

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:  # Sentinel to shutdown
                break
            file_path, embeddings = item
            try:
                np.save(file_path, embeddings)
            except Exception as e:
                logger.error(f"Error saving {file_path}: {e}")
            self.queue.task_done()

    def save(self, file_path: str, embeddings: np.ndarray):
        self.queue.put((file_path, embeddings))

    def close(self):
        # Signal the worker to shutdown and wait for it to finish.
        self.queue.put(None)
        self.thread.join()


class AsyncChunkIDSaver:
    """
    Saves data to disk asynchronously using a background thread.
    """

    def __init__(self, max_queue_size: int = 3):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True  # Allow thread to exit if main process exits
        self.thread.start()

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:  # Sentinel to shutdown
                break
            file_path, chunk_ids = item
            try:
                np.save(file_path, chunk_ids)
            except Exception as e:
                logger.error(f"Error saving {file_path}: {e}")
            self.queue.task_done()

    def save(self, file_path: str, chunk_ids: np.ndarray):
        self.queue.put((file_path, chunk_ids))

    def close(self):
        # Signal the worker to shutdown and wait for it to finish.
        self.queue.put(None)
        self.thread.join()
