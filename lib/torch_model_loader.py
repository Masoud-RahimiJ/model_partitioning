import itertools
from threading import Event
from lib.model_loader import ModelLoader
from torch import load

class TorchModelLoader(ModelLoader):
    def __init__(self, model_initializer_fn, s3_bucket, config):
        super().__init__(model_initializer_fn, s3_bucket, config)
        
    def _wrap_model(self, model):
        wrap_module(model)
        
    def _load_partition(self, partition, partition_name):
        partition_state_dict = load(partition)
        if not self._model_initialized_event.is_set():
            self._model_initialized_event.wait()
        self._model.load_state_dict(partition_state_dict, strict=False)
        
        

def extract_module_params(module):
    persistent_buffers = {k: v for k, v in module._buffers.items() if k not in module._non_persistent_buffers_set}
    local_name_params = itertools.chain(module._parameters.items(), persistent_buffers.items())
    return {k: v for k, v in local_name_params if v is not None}

def wrap_param_copy(param, cp):
    def wrapped_function(input_param, non_blocking=False):
        result = cp(input_param, non_blocking)
        param.is_loaded = True
        return result
    return wrapped_function
    
def load_state_dict_post_hook(module, _):
    if not module.is_loaded.is_set():
        params = extract_module_params(module)
        for _, param in params.items():
            if param.is_loaded == False: return
        module.is_loaded.set()
    
def forward_pre_hook(module, _):
    if not module.is_loaded.is_set():
        module.is_loaded.wait()
    print("n")
        
def wrap_layer(module):
    params = extract_module_params(module)
    if len(params) > 0:
        for _, param in params.items():
            param.is_loaded = False
            param.copy_ = wrap_param_copy(param, param.copy_)
        module.is_loaded = Event()
        module.register_forward_pre_hook(forward_pre_hook)
        module.register_load_state_dict_post_hook(load_state_dict_post_hook)
 
def wrap_module(model):
    wrap_layer(model)
    for module in model.children():
        wrap_module(module)
