import numpy as np
import onnx
from onnxconverter_common import float16, auto_convert_mixed_precision

original_model_path = "models/inswapper_128.onnx"
fp16_model_path = "models/inswapper_128_fp16_new.onnx"


model = onnx.load(original_model_path)
#model_fp16 = float16.convert_float_to_float16(model, keep_io_types = True)

source_tensor = np.random.rand(1, 512).astype(np.float32)
target_tensor = np.random.rand(1, 3, 128, 128).astype(np.float32)

feed_dict =\
{
    'source': source_tensor,
    'target': target_tensor
}

model_fp16 = auto_convert_mixed_precision(model, feed_dict, rtol=0.01, atol=0.001, keep_io_types=True)
onnx.save(model_fp16, fp16_model_path)
