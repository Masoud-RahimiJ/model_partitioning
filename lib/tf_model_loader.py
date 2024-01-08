from threading import Event
from lib.model_loader import ModelLoader
import os, time


class TFModelLoader(ModelLoader):
    def __init__(self, model_initializer_fn, s3_bucket, config):
        super().__init__(model_initializer_fn, s3_bucket, config)
        
    def _wrap_model(self, model):
        wrap_module(model)
        
    def _load_partition(self, partition, partition_name):
        try:
            # with open(partition_name, 'wb') as f:
            #     f.write(partition.read())
            if not self._model_initialized_event.is_set():
                self._model_initialized_event.wait()
            self._model.load_weights(partition_name, by_name=True, skip_mismatch=True)
            os.remove(partition_name)
        except Exception as e:
            print(1)
            print(e)


def wrap_module(model):
    wrap_layer(model)
    for module in getattr(model, "layers", []):
        wrap_module(module)
        
def wrap_layer(module):
    params = extract_module_params(module)
    if len(params) > 0:
        for param in params:
            param.is_loaded = False
            if hasattr(param, '_assign_placeholder'):
                param._assign_op = wrap_param_assign_op(param, param._assign_op)
            else:
                param.assign = wrap_param_assign(param, param.assign)
        module.is_loaded = Event()
        module.call = wrap_module_call(module, module.call)
        module.finalize_state = wrap_module_finalize_state(module, module.finalize_state)

def extract_module_params(module):
    return module.trainable_weights

def wrap_param_assign_op(param, assign):
    def wrapped_function(input_param, use_locking=False):
        result = assign(input_param, use_locking)
        param.is_loaded = True
        return result
    return wrapped_function

def wrap_param_assign(param, assign):
    def wrapped_function(shape):
        assign_f = assign(shape)
        param.is_loaded = True
        return assign_f
    return wrapped_function

def wrap_module_finalize_state(module, finalize_state):
    def wrapped_finalize_state():
        if not module.is_loaded.is_set():
            params = extract_module_params(module)
            for param in params:
                if param.is_loaded == False: return
            module.is_loaded.set()
        finalize_state()
    return wrapped_finalize_state
    
def wrap_module_call(module, call):
    def wrapped_call(*args, **kwargs):
        if not module.is_loaded.is_set():
            module.is_loaded.wait()
        return call(args[0])
    return wrapped_call
        