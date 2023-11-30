from concurrent import futures
from threading import Lock, Event, Thread
import io

CHUNK_SIZE = 1024 * 1024
COUNT_DOWNLOAD_THREADS = 2
COUNT_LOAD_THREADS = 2

class ModelLoader:
    def __init__(self, model_initializer_fn, s3_bucket, config):
        self._model_initializer_fn = model_initializer_fn
        self._s3_bucket = s3_bucket
        self._download_delay = config.get('download_delay')
        self._partition_names = config.get('partition_names')
        self._model = None
        self._load_thread_pool = futures.ThreadPoolExecutor(max_workers=COUNT_LOAD_THREADS)
        self._download_lock = Lock()
        self._model_initialized_event = Event()
        
    def load(self):
        t1 = Thread(target=self._load_model)
        t1.start()
        model = self._model_initializer_fn()
        self._wrap_model(model)
        self._model = model
        self._model_initialized_event.set()
        return self._model
        
    def _load_model(self):
        with futures.ThreadPoolExecutor(max_workers=COUNT_DOWNLOAD_THREADS) as executor:
            [executor.submit(self._download_and_load_partition, partition_name) for partition_name in self._partition_names]
        self._load_thread_pool.shutdown(wait=True)
            
    def _load_partition(self, partition, partition_name):
        raise NotImplementedError()
    
    def _wrap_model(self, model):
        raise NotImplementedError()
          
    def _download_and_load_partition(self, partition_name):
        print(partition_name)
        partition_data = io.BytesIO()
        parition_obj = self._s3_bucket.Object(partition_name)
        partition_length = parition_obj.content_length
        partition_body = parition_obj.get()['Body']
        download_stream = partition_body.iter_chunks(CHUNK_SIZE)
        is_locked = True
        self._download_lock.acquire()
        for chunk in download_stream:
            if partition_length - partition_body.tell() < self._download_delay and is_locked:
                self._download_lock.release()
                is_locked = False
            partition_data.write(chunk)
        partition_data.seek(0)
        self._load_thread_pool.submit(self._load_partition, partition_data, partition_name)
        