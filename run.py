#!/usr/bin/env python3
import sys

import cv2
import dlib
import numpy
import onnx
import onnxruntime
from onnx import numpy_helper
from argparse import ArgumentParser


# Create an argument parser
parser = ArgumentParser()
parser.add_argument('--use-fp16', action = 'store_true')

# Parse the command-line arguments
args = parser.parse_args()

# Access the selected model path via args.model_path
SWAPPER_MODEL_PATH = 'models/inswapper_128_fp16.onnx' if args.use_fp16 else 'models/inswapper_128.onnx'

INSWAPPER_MATRIX = numpy_helper.to_array(onnx.load(SWAPPER_MODEL_PATH).graph.initializer[-1])
INSWAPPER = onnxruntime.InferenceSession(SWAPPER_MODEL_PATH, providers=[ 'CPUExecutionProvider' ])
ARCFACE = onnxruntime.InferenceSession('models/arcface_w600k_r50.onnx', providers=[ 'CPUExecutionProvider' ])
FACE_DETECTOR = dlib.get_frontal_face_detector()
SHAPE_PREDICTOR = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
TEMPLATES =\
{
	'arcface_112_v2': numpy.array(
	[
		[0.34191607, 0.46157411],
		[0.65653393, 0.45983393],
		[0.50022500, 0.64050536],
		[0.37097589, 0.82469196],
		[0.63151696, 0.82325089]
	]),
	'arcface_128_v2': numpy.array(
	[
		[0.36167656, 0.40387734],
		[0.63696719, 0.40235469],
		[0.50019687, 0.56044219],
		[0.38710391, 0.72160547],
		[0.61507734, 0.72034453]
	]),
}


def get_kps(img):
	image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	face = FACE_DETECTOR(image_gray)[0]
	shape = SHAPE_PREDICTOR(image_gray, face)
	left_eye = numpy.mean(shape.parts()[36:42], axis=0)
	right_eye = numpy.mean(shape.parts()[42:48], axis=0)
	nose = shape.part(30)
	left_mouth = shape.part(48)
	right_mouth = shape.part(54)
	kps = [left_eye, right_eye, nose, left_mouth, right_mouth]
	kps = numpy.array([[lmk.x, lmk.y] for lmk in kps]).astype('float32')
	return kps


def warp_face_by_kps(temp_frame, kps, template, crop_size):
	normed_template = TEMPLATES.get(template) * crop_size
	affine_matrix = cv2.estimateAffinePartial2D(kps, normed_template, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
	crop_frame = cv2.warpAffine(temp_frame, affine_matrix, crop_size, borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
	return crop_frame, affine_matrix


def paste_back(temp_frame, crop_frame, crop_mask, affine_matrix):
	inverse_matrix = cv2.invertAffineTransform(affine_matrix)
	temp_frame_size = temp_frame.shape[:2][::-1]
	inverse_crop_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_frame_size).clip(0, 1)
	inverse_crop_frame = cv2.warpAffine(crop_frame, inverse_matrix, temp_frame_size, borderMode=cv2.BORDER_REPLICATE)
	paste_frame = temp_frame.copy()
	paste_frame[:, :, 0] = inverse_crop_mask * inverse_crop_frame[:, :, 0] + (1 - inverse_crop_mask) * temp_frame[:, :, 0]
	paste_frame[:, :, 1] = inverse_crop_mask * inverse_crop_frame[:, :, 1] + (1 - inverse_crop_mask) * temp_frame[:, :, 1]
	paste_frame[:, :, 2] = inverse_crop_mask * inverse_crop_frame[:, :, 2] + (1 - inverse_crop_mask) * temp_frame[:, :, 2]
	return paste_frame


def calc_embedding(temp_frame, kps):
	crop_frame, matrix = warp_face_by_kps(temp_frame, kps, 'arcface_112_v2', (112, 112))
	crop_frame = crop_frame.astype(numpy.float32) / 127.5 - 1
	crop_frame = crop_frame[:, :, ::-1].transpose(2, 0, 1)
	crop_frame = numpy.expand_dims(crop_frame, axis=0)
	embedding = ARCFACE.run(None,
	{
		ARCFACE.get_inputs()[0].name: crop_frame
	})[0]
	embedding = embedding.ravel()
	normed_embedding = embedding / numpy.linalg.norm(embedding)
	return embedding, normed_embedding


def swap_face(source, target):
	source_kps = get_kps(source)
	target_kps = get_kps(target)
	source_embedding = calc_embedding(source, source_kps)[1].reshape((1, -1))
	source_embedding = numpy.dot(source_embedding, INSWAPPER_MATRIX) / numpy.linalg.norm(source_embedding)
	crop_frame, affine_matrix = warp_face_by_kps(target, target_kps, 'arcface_128_v2', (128, 128))
	crop_frame = (crop_frame[:, :, ::-1] / 255.0).transpose(2, 0, 1).astype(numpy.float32)
	crop_frame = INSWAPPER.run(None,
	{
		'source': source_embedding,
		'target': [crop_frame]
	})[0][0]
	crop_frame = (crop_frame.transpose(1, 2, 0) * 255.0).round()[:, :, ::-1].astype(numpy.uint8)
	crop_mask = numpy.zeros(crop_frame.shape[:2], dtype=numpy.float32)
	crop_mask = cv2.rectangle(crop_mask, (8, 8), (120, 120), 255, -1) / 255.0
	crop_mask = cv2.GaussianBlur(crop_mask, (15, 15), 0)
	paste_frame = paste_back(target, crop_frame, crop_mask, affine_matrix)
	return paste_frame


if __name__ == '__main__':
	source = cv2.imread('examples/source.jpg')
	target = cv2.imread('examples/target.jpg')
	output = swap_face(source, target)
	cv2.imshow('output', output)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	sys.exit()
