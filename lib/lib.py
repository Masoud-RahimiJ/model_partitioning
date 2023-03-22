import itertools
from threading import Semaphore
from torch.nn import Module

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
        params = extract_module_params(module)
        for _, param in params.items():
            if param.is_loaded == False: return
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
        if getattr(module.__class__, "set_extra_state", Module.set_extra_state) is not Module.set_extra_state:
            print("eeeeeeeeee")

def wrap_model(model):
    wrap_module(model)
    for module in model.children():
        wrap_model(module)