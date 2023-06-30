import torch
import onnx
import onnx_tf

def convert(model, filename):
    torch.onnx.export(model,               # model being run
                  torch.randn(1, 3, 224, 224), # dummy input (required)
                  "temp.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True)
    onnx_model = onnx.load("temp.onnx")
    tf_model = onnx_tf.convert_from_onnx(onnx_model)
    
    