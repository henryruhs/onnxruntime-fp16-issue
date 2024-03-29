> The converted 16 floating point version of the `inswapper_128.onnx` model does not work since `onnxruntime==1.17.0` - it seems to be broken for CPU and CUDA but works for TensorRT.

# Problem

The lack of FP16 supports causes different results depending on the integration:

## Distorted face (cuda)

![Broken1](https://raw.githubusercontent.com/henryruhs/onnxruntime-fp16-issue/master/examples/output-broken1.jpg?sanitize=true)

## Face box being black (cpu)

![Broken2](https://raw.githubusercontent.com/henryruhs/onnxruntime-fp16-issue/master/examples/output-broken2.jpg?sanitize=true)


# Installation

## 1. Download the models

```
curl --create-dirs --insecure --location --continue-at - --output ./models/inswapper_128.onnx https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx
curl --create-dirs --insecure --location --continue-at - --output ./models/inswapper_128_fp16.onnx https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx
curl --create-dirs --insecure --location --continue-at - --output ./models/arcface_w600k_r50.onnx https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_w600k_r50.onnx
curl --create-dirs --insecure --location --continue-at - --output ./models/shape_predictor_68_face_landmarks.dat https://github.com/tzutalin/dlib-android/raw/master/data/shape_predictor_68_face_landmarks.dat
```

## 2. Use VENV

```
python3.10 -m venv venv
```
```
source venv/bin/activate
```

## 3. Install dependencies

```
pip install -r requirements.txt
```

## 4. Run

Run float:

```
python run.py
```

Run float 16:

```
python run.py --use-fp16
```


# Replicate

Feel free to convert it yourself using `convert_fp16.py`. I had to disable the 3 x validate exceptions inside `venv/lib/python3.10/site-packages/onnxconverter_common/auto_mixed_precision.py` code to make it pass.
