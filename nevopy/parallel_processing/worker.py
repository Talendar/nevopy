"""
TODO
"""

import multiprocessing
import threading
import os
from datetime import datetime
from timeit import default_timer as timer
from enum import Enum


class LocalWorker:
    """
    TODO
    """

    def __init__(self, worker_id, num_processes=None):
        """
        TODO

        :param worker_id:
        :param num_processes:
        """
        self._wid = worker_id
        self._lock = multiprocessing.Lock()
        self.error_count = 0
        self.__status = WorkerStatus.READY
        self._pool = multiprocessing.Pool(processes=os.cpu_count() if num_processes is None else num_processes)

        self.history = []
        self.work_session_info = None
        self._work_timer_start = None

        self._user_callback = None
        self._user_error_callback = None
        self._batch = None

    @property
    def id(self):
        return self._wid

    def check_status(self, status):
        """ Safely checks if the worker is currently in the given state. """
        with self._lock:
            return self.__status == status

    def _set_status(self, status):
        """ Safely updates the current state of the worker. """
        with self._lock:
            self.__status = status

    def terminate(self):
        """ Terminates the worker, freeing its resources. """
        self._pool.terminate()
        self._pool.close()

    def _work_finished(self, error_occurred):
        """ Updates the worker history and current state. """
        self.work_session_info.processing_time = timer() - self._work_timer_start
        self.work_session_info.end_time = datetime.now()
        self.work_session_info.error = error_occurred

        self.history.append(self.work_session_info)
        self._set_status(WorkerStatus.READY if not error_occurred else WorkerStatus.ERROR)

    def _callback(self, results):
        """ Called when the process pool successfully finishes the work. """
        self._work_finished(error_occurred=False)
        if self._user_callback is not None:
            self._user_callback(self, self._batch, results)

    def _error_callback(self, result):
        """ Called when the process pool finishes the work with an error. """
        self._work_finished(error_occurred=True)
        if self._user_error_callback is not None:
            self._user_error_callback(self, self._batch)

    def _thread_work(self, func):
        results = self._pool.map(func=func, iterable=self._batch.items)
        self._callback(results)

    def work(self, batch, func, callback, error_callback):
        """
        TODO

        :param func:
        :param batch:
        :param callback:
        :param error_callback:
        :return:
        """
        # checking if the worker is busy
        if self.check_status(WorkerStatus.BUSY):
            raise WorkerBusyException("Attempt to assign new work to a busy worker!")
        self._set_status(WorkerStatus.BUSY)

        # setting callbacks
        self._user_callback = callback
        self._user_error_callback = error_callback
        self._batch = batch

        # preparing history info
        self.work_session_info = WorkSessionInfo()
        self._work_timer_start = timer()

        # assigning work
        """self._pool.map_async(func=func,
                             iterable=self._batch.items,
                             callback=self._callback,
                             error_callback=self._error_callback)"""
        thread = threading.Thread(target=self._thread_work, args=(func,))
        thread.start()


class WorkSessionInfo:
    """
    TODO
    """

    def __init__(self):
        """
        TODO
        """
        self.start_time = datetime.now()
        self.end_time = None
        self.error = None
        self.processing_time = None


class WorkBatch:
    """ Represents a batch of items still to be processed. """
    def __init__(self, bid, items):
        self.id = bid
        self.items = items


class WorkerStatus(Enum):
    READY, BUSY, ERROR = range(3)


class WorkerBusyException(RuntimeError):
    """ Raised to indicate that a worker is currently busy (working). """
    pass
