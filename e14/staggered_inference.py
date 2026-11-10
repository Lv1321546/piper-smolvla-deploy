"""Bounded, staggered requests to identical single-GPU policy replicas."""
import concurrent.futures
import math

from e14.inference_client import InferenceClient
from shared.deployment import check_model
from shared.protocol import finite_number


class StaggeredInference:
    def __init__(self, config):
        options = config['client']
        urls = options.get('replica_urls', [options['url']])
        if not isinstance(urls, list) or not urls:
            raise ValueError('client.replica_urls must be a nonempty list')
        self.clients = [InferenceClient({**config, 'client': {**options, 'url': url}}) for url in urls]
        self.urls = [client.url for client in self.clients]
        if len(set(self.urls)) != len(self.urls) or self.urls[0] != options['url'].rstrip('/'):
            raise ValueError('Replica URLs must be unique and start with client.url')
        self.config = config
        self.interval = finite_number(options['inference_interval_steps'], 'inference_interval_steps',
                                      positive=True) / config['robot']['control_hz']
        self.next_dispatch = -math.inf
        self.not_before = [-math.inf] * len(self.clients)
        self.pending = [None] * len(self.clients)
        self.cursor = 0
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.clients),
                                                         thread_name_prefix='policy-http')

    def health(self, require_executable=False):
        metadata = [check_model(client.health(require_executable), self.config,
                                physical=require_executable) for client in self.clients]
        if any(item != metadata[0] for item in metadata[1:]):
            raise ValueError('Inference replicas advertise different model metadata')
        return metadata[0]

    def _predict(self, index, observation, task, require_executable):
        result = self.clients[index].predict(observation, task, None, require_executable)
        return dict(result, replica_index=index, replica_url=self.urls[index])

    def submit(self, observation, task, now, require_executable=False):
        if now < self.next_dispatch:
            return None
        for offset in range(len(self.clients)):
            index = (self.cursor + offset) % len(self.clients)
            if self.pending[index] is None and now >= self.not_before[index]:
                self.pending[index] = self.pool.submit(self._predict, index, observation, task,
                                                       require_executable)
                self.cursor = (index + 1) % len(self.clients)
                self.next_dispatch = now + self.interval
                return index
        # No request queue: the next control tick supplies a newer observation.
        return None

    def completed(self):
        completed = []
        for index, future in enumerate(self.pending):
            if future is not None and future.done():
                self.pending[index] = None
                completed.append((index, future))
        return completed

    def defer(self, index, until):
        self.not_before[index] = until

    def close(self):
        for future in self.pending:
            if future is not None:
                future.cancel()
        # Called after the actuator hold. HTTP requests have an absolute timeout.
        self.pool.shutdown(wait=True, cancel_futures=True)
