from threading import Event

def extract_module_params(module):
    module.trainable_weights()

def wrap_param_assign(param, assign):
    def wrapped_function(input_param, use_locking=False):
        result = assign(input_param, use_locking)
        param.is_loaded = True
        return result
    return wrapped_function

def wrap_module_finalize_state(module, finalize_state):
    def wrapped_finalize_state():
        if not module.is_loaded.is_set():
            params = extract_module_params(module)
            for param in params.items():
                if param.is_loaded == False: return
            module.is_loaded.set()
        finalize_state()
    return wrapped_finalize_state
    
def wrap_module_call(module, call):
    def wrapped_call(*args, **kwargs):
        if not module.is_loaded.is_set():
            module.is_loaded.wait()
        call(args, kwargs)
    return wrapped_call
        
def wrap_layer(module):
    params = extract_module_params(module)
    if len(params) > 0:
        for param in params:
            param.is_loaded = False
            if hasattr(param, '_assign_placeholder'):
                param._assign_op = wrap_param_assign(param, param._assign_op)
            else:
                param.assign = wrap_param_assign(param, param.assign)
        module.is_loaded = Event()
        wrap_module_call(module, module.__call__)
        wrap_module_finalize_state(module, module.finalize_state)

def wrap_module(model):
    wrap_layer(model)
    for module in getattr(model, "layers", []):
        wrap_module(module)