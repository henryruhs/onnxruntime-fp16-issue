> The converted version of the inswapper model does not work with onnxruntime 1.17.0 - it does not depend on GPU or CPU.


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
