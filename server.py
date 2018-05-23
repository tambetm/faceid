from flask import Flask, request, jsonify
import facedb as db
import faceid
import os
import time
import argparse

other_args = faceid.parse_args()
parser = argparse.ArgumentParser()
parser.add_argument('--host', default='0.0.0.0')
parser.add_argument('--port', type=int, default=9014)
args = parser.parse_args(other_args)

faceid.init()

app = Flask(__name__)

@app.route("/scan", methods=['POST'])
def scan():
    params = request.get_json()
    data_dir = params['data_dir']
    dbfile = params['dbfile']
    files = params['files']

    start_time = time.time()
    db.connect(os.path.join(data_dir, dbfile))
    num_files = 0
    num_images = 0
    num_faces = 0
    for file in files:
        filepath = os.path.join(data_dir, file['filename'])
        file_images, file_faces = faceid.process_file(filepath)
        num_files += 1
        num_images += file_images
        num_faces += file_faces
    db.close()
    elapsed = time.time() - start_time

    res = {'num_files': num_files, 'num_images': num_images, 'num_faces': num_faces, 'elapsed': elapsed, 'images_per_s': num_images / elapsed}
    return jsonify(res)

@app.route("/compute_similarities", methods=['POST'])
def compute_similarities():
    params = request.get_json()
    data_dir = params['data_dir']
    dbfile = params['dbfile']

    start_time = time.time()
    db.connect(os.path.join(data_dir, dbfile))
    num_faces, num_similarities = faceid.compute_similarities()
    db.close()
    elapsed = time.time() - start_time

    res = {'num_similarities': num_similarities, 'num_faces': num_faces, 'elapsed': elapsed}
    return jsonify(res)

if __name__ == '__main__':
    app.run(args.host, args.port)
