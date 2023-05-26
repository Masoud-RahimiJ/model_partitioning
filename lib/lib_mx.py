from threading import Event

def extract_module_params(module):
    return module._reg_params.items()

def wrap_param_load_init(param, load_init):
    def wrapped_function(param, device, cast_dtype=False, dtype_source=False):
        result = load_init(param, device, cast_dtype, dtype_source)
        param.is_loaded = True
        return result
    return wrapped_function

def check_module_loading_state(module):
    for layer in module._children.values():
        check_module_loading_state(layer)
    if getattr(module, 'is_loaded')is not None and not module.is_loaded.is_set():
            params = extract_module_params(module)
            for param in params:
                if param.is_loaded == False: return
            module.is_loaded.set()
    
def wrap_module_load_dict(module, load_dict):
    def wrapped_load_dict(self, param_dict, device=None, allow_missing=False,
                  ignore_extra=False, cast_dtype=False, dtype_source="current"):
        load_dict(self, param_dict, device, allow_missing,
                  ignore_extra, cast_dtype, dtype_source)
        check_module_loading_state(module)
    return wrapped_load_dict
    
def forward_pre_hook(block, _):
    if not block.is_loaded.is_set():
        block.is_loaded.wait()
        
def wrap_layer(module):
    params = extract_module_params(module)
    if len(params) > 0:
        for param in params:
            param.is_loaded = False
            param._load_init = wrap_param_load_init(param, param._load_init)
        module.is_loaded = Event()
        module.register_forward_pre_hook(forward_pre_hook)
        wrap_module_load_dict(module, module.load_dict)

def wrap_module(model):
    wrap_layer(model)
    for module in model._children.values():
        wrap_module(module)