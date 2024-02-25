from threading import Event
from lib.model_loader import ModelLoader
import os, time
from tensorflow.python.keras.saving.hdf5_format import load_attributes_from_hdf5_group
from tensorflow.python.keras import backend
import h5py
import numpy as np
import tensorflow as tf

class TFModelLoader(ModelLoader):
    def __init__(self, model_initializer_fn, s3_bucket, config):
        super().__init__(model_initializer_fn, s3_bucket, config)
        
    def _wrap_model(self, model):
        self.prams_dict = wrap_module(model)
        
    def _load_partition(self, partition, partition_name):
        try:
            # with open(partition_name, 'wb') as f:
            #     f.write(partition.read())
            if not self._model_initialized_event.is_set():
                self._model_initialized_event.wait()
            # self.load_partition_tf(partition_name)
            self._model.load_weights(partition_name, by_name=True, skip_mismatch=True)
            os.remove(partition_name)
        except Exception as e:
            print("!!!!!!")
            print(e)
            
    def load_partition_tf(self, partition_name):
        weight_value_tuples = []
        f = h5py.File(partition_name, "r")
        for name in load_attributes_from_hdf5_group(f, 'layer_names'):
            g = f[name]
            for w in load_attributes_from_hdf5_group(g, 'weight_names'):
                if '/'.join(w.split('/')[2:]) in self.prams_dict:
                    weight_value_tuples.append((self.prams_dict['/'.join(w.split('/')[2:])], np.asarray(g[w])))
        f.close()
        with tf.init_scope():
            backend.batch_set_value(weight_value_tuples)
        for m in self._model._flatten_layers():
            m.finalize_state()
            
        


def wrap_module(model):
    prams_dict = {}
    for m in model._flatten_layers():
        wrap_layer(m, prams_dict)
    return prams_dict

def wrap_layer(module, prams_dict):
    params = extract_module_params(module)
    if len(params) > -1 and module.name in ["input_1", "conv4_block10_2_bn", "conv4_block22_1_bn", "conv5_block3_1_bn"]:
        print("yes")
        for param in params:
            prams_dict['/'.join(param.name.split('/')[1:])] = param
            param.is_loaded = False
            if hasattr(param, '_assign_placeholder'):
                param._assign_op = wrap_param_assign_op(param, param._assign_op)
            else:
                param.assign = wrap_param_assign(param, param.assign)
        module.is_loaded = Event()
        module.call = wrap_module_call(module, module.call)
        module.finalize_state = wrap_module_finalize_state(module, module.finalize_state)

def extract_module_params(module):
    return module._trainable_weights

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
        return call(*args, **kwargs)
    return wrapped_call
        
        
        


