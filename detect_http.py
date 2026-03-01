import tensorflow as tf
import numpy as np
from PIL import Image
import io
import argparse
from flask import Flask, request, jsonify
import logging
from logging.handlers import SysLogHandler

classes = ['Burn', 'Crack', 'Delamination', 'Dust', 'Normal']

app = Flask(__name__)


def setup_logging():
	logger = logging.getLogger('detect')
	logger.setLevel(logging.INFO)
	fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

	# Console handler
	ch = logging.StreamHandler()
	ch.setFormatter(fmt)
	logger.addHandler(ch)

	# Attempt to add a SysLogHandler (best-effort; works if syslog is reachable)
	try:
		sh = SysLogHandler(address=('localhost', 514))
		sh.setFormatter(fmt)
		logger.addHandler(sh)
		logger.info('Syslog handler attached')
	except Exception:
		logger.info('Syslog handler not attached (optional)')

	return logger


logger = setup_logging()


def load_model_safe(path='solar_fault_model.h5'):
	try:
		m = tf.keras.models.load_model(path)
		logger.info(f'Model loaded from {path}')
		return m
	except Exception as e:
		logger.exception(f'Failed to load model from {path}: {e}')
		raise


model = load_model_safe('solar_fault_model.h5')


def predict_image(pil_img):
	img = pil_img.convert('RGB').resize((224, 224))
	input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
	prediction = model.predict(input_data)
	probs = prediction.flatten().tolist()
	label = classes[int(np.argmax(prediction))]
	return label, probs


@app.before_request
def log_request_info():
	try:
		addr = request.remote_addr or 'unknown'
		logger.info(f'Incoming {request.method} {request.path} from {addr} Content-Length={request.content_length}')
	except Exception:
		logger.exception('Failed to log request info')


@app.route('/detect', methods=['POST'])
def detect_endpoint():
	# Accept either multipart file 'image' or raw JPEG bytes
	try:
		if 'image' in request.files:
			file = request.files['image']
			img_bytes = file.read()
			logger.info(f'Received multipart image; filename={file.filename} size={len(img_bytes)}')
		else:
			img_bytes = request.get_data()
			logger.info(f'Received raw image data size={len(img_bytes) if img_bytes else 0}')
		if not img_bytes:
			logger.warning('No image provided in request')
			return jsonify({'error': 'No image provided'}), 400

		try:
			pil_img = Image.open(io.BytesIO(img_bytes))
		except Exception as e:
			logger.exception('Invalid image received')
			return jsonify({'error': 'Invalid image', 'detail': str(e)}), 400

		label, probs = predict_image(pil_img)
		logger.info(f'Prediction: {label} probs={[round(p,4) for p in probs]}')
		return jsonify({'label': label, 'probabilities': probs})
	except Exception as e:
		logger.exception('Error processing /detect')
		return jsonify({'error': 'Internal server error', 'detail': str(e)}), 500


def run_server(host='0.0.0.0', port=5000):
	logger.info(f'Starting detection server on {host}:{port}')
	# Disable reloader to avoid duplicate logs/instances
	app.run(host=host, port=port, use_reloader=False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run detection server or single-image prediction')
	parser.add_argument('--server', action='store_true', help='Run HTTP server to accept images')
	parser.add_argument('--image', type=str, help='Path to image file for local prediction')
	args = parser.parse_args()

	if args.server:
		run_server()
	elif args.image:
		pil = Image.open(args.image)
		label, probs = predict_image(pil)
		print('Detected:', label)
		print('Probabilities:', probs)
	else:
		print('No action specified. Use --server to run HTTP server or --image <path> to test a single image.')
