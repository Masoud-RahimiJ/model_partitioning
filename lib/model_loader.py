from concurrent import futures
from threading import Lock, Event, Thread
import io, time

CHUNK_SIZE = 1024 * 1024 * 100
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
        self.tmp = []
        
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
            stt=time.time()
            [executor.submit(self._download_and_load_partition, partition_name) for partition_name in self._partition_names]
            executor.shutdown(wait=True)
            print("download: ", time.time()-stt)
        self._load_thread_pool.shutdown(wait=True)
        print("load: ", sum(self.tmp))
            
    def _load_partition(self, partition, partition_name):
        raise NotImplementedError()
    
    def _wrap_model(self, model):
        raise NotImplementedError()
    
    # def _download_and_load_partition(self, partition_name):
        # partition_data = io.BytesIO()
        # parition_obj = self._s3_bucket.Object(partition_name)
        # self._download_lock.acquire()
        # data = io.BytesIO(parition_obj.get()['Body'].read())
        # self._download_lock.release()
        # self._load_thread_pool.submit(self._load_partition, data, partition_name)
        
          
    def _download_and_load_partition(self, partition_name):
        partition_data = io.BytesIO()
        parition_obj = self._s3_bucket.Object(partition_name)
        partition_length = parition_obj.content_length
        partition_body = parition_obj.get()['Body']
        chunks_size = CHUNK_SIZE
        # if partition_length > 2 * self._download_delay:
        #     chunks_size = int(partition_length - self._download_delay / 10)
        download_stream = partition_body.iter_chunks(chunks_size)
        is_locked = True
        self._download_lock.acquire()
        aaa=time.time()
        for chunk in download_stream:
            if is_locked and partition_length - partition_body.tell() <= self._download_delay:
                print(partition_length - partition_body.tell())
                self._download_lock.release()
                is_locked = False
            partition_data.write(chunk)
        print(time.time()-aaa, partition_length/1000000)
        
        partition_data.seek(0)
        self._load_thread_pool.submit(self._load_partition, partition_data, partition_name)
        