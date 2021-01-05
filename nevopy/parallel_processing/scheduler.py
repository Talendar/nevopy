"""

"""

import time
import numpy as np
from nevopy.parallel_processing.worker import *


class JobScheduler:
    """
    TODO
    """

    def __init__(self,
                 worker_list,
                 error_tolerance=3,
                 worker_wait_time=0.1,
                 work_distribution_wait_time=0.1):
        """
        TODO
        :param worker_list:
        """
        self._error_tolerance = error_tolerance
        self._worker_wait_time = worker_wait_time
        self._work_distribution_wait_time = work_distribution_wait_time

        self._workers = {w.id: w for w in worker_list}
        self._idle_workers = worker_list[:]

        self._lock = multiprocessing.Lock()
        self._all_batches = None
        self._batches_queue = None
        self._batches_results = None

    def terminate_all(self):
        """ Terminates all the workers used by the scheduler. """
        for worker in self._workers.values():
            worker.terminate()

    def _next_batch(self):
        """ Returns the next batch in the processing queue. """
        with self._lock:
            if len(self._batches_queue) > 0:
                return self._batches_queue.pop(0)
            return None

    def _get_idle_worker(self):
        """ Returns a worker if an idle one is find or None if there is no idle worker. """
        with self._lock:
            if len(self._idle_workers) > 0:
                return self._idle_workers.pop()
            return None

    def _worker_callback(self, worker, batch, results):
        """ Called by the worker when it finishes its job without error. """
        with self._lock:
            self._batches_results[batch.id] = results
            self._idle_workers.append(worker)

    def _worker_error_callback(self, worker, batch):
        """ Called by the worker when it encounters an error during its execution. """
        with self._lock:
            self._batches_queue.append(batch)

        worker.error_count += 1
        if worker.error_count > self._error_tolerance:
            self._workers.pop(worker.id)
            worker.terminate()
        else:
            with self._lock:
                self._idle_workers.append(worker)

    def _distribute_work(self, func):
        """ Distributes the processing of the current batches among the available workers. """
        if len(self._workers) == 0:
            raise NoWorkersError("The scheduler has no workers available!")

        while (batch := self._next_batch()) is not None:
            # trying to assign the batch to a worker
            if (worker := self._get_idle_worker()) is not None:
                worker.work(batch=batch, func=func,
                            callback=self._worker_callback,
                            error_callback=self._worker_error_callback)
            # no idle worker available; sleeping for a while before trying again
            else:
                time.sleep(self._worker_wait_time)

    def _jobs_done(self):
        """ Returns True if all the batches have been processed. """
        for batch in self._all_batches:
            if batch.id not in self._batches_results:
                return False
        return True

    def run(self, items, func):
        """

        :param items:
        :param func:
        :return:
        """
        # todo: uneven split (allow the creation of batches of different sizes, so that different workers can have a
        # todo: different number of items to process).
        self._batches_queue = [WorkBatch(bid=i, items=b)
                               for i, b in enumerate(np.split(np.array(items), len(self._workers)))]
        self._all_batches = self._batches_queue[:]
        self._batches_results = {}

        while True:
            self._distribute_work(func)
            if self._jobs_done():
                break
            time.sleep(self._work_distribution_wait_time)

        results = []
        for batch in self._all_batches:
            results += self._batches_results[batch.id]

        assert len(items) == len(results)
        return results


class NoWorkersError(Exception):
    """ Raised when a scheduler has no worker. """
    pass
