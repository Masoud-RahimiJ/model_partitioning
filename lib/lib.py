import itertools
from threading import Semaphore

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
    if getattr(module, "must_be_loaded", False):
        for buf in module.buffers(False):
            if buf is not None and not getattr(buf, "is_loaded", True):
                return
        for param in module.parameters(False):
            if param is not None and not getattr(param, "is_loaded", True):
                return
        module.must_be_loaded = False
        module.is_loaded_lock.release()
    
def forward_pre_hook(module, _):
    if getattr(module, "must_be_loaded", False) :
        module.is_loaded_lock.acquire()
        module.is_loaded_lock.release()
        
def wrap_module(module):
    params = extract_module_params(module)
    if len(params) > 0:
        for _, param in params.items():
            param.is_loaded = False
            param.copy_ = wrap_param_copy(param, param.copy_)
        module.must_be_loaded = True
        module.is_loaded_lock = Semaphore(0)
        module.register_forward_pre_hook(forward_pre_hook)
        module.register_load_state_dict_post_hook(load_state_dict_post_hook)

def wrap_model(model):
    wrap_module(model)
    for module in model.children():
        wrap_model(module)