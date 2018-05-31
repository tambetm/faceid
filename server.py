from flask import Flask, request, jsonify
import facedb as db
import faceid
import os
import time
import argparse

parser = argparse.ArgumentParser(parents=[faceid.argparser()])
parser.add_argument('--host', default='0.0.0.0')
parser.add_argument('--port', type=int, default=9014)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--debug_server', action='store_true')
args = parser.parse_args()

faceid.init(args)

app = Flask(__name__)

@app.route("/scan", methods=['POST'])
def scan():
    params = request.get_json()
    data_dir = params['data_dir']
    dbfile = params['dbfile']
    source = params.get('source', args.source)
    files = params['files']

    if faceid.args.save_resized:
        faceid.makedirs(os.path.join(data_dir, faceid.args.save_resized))

    if faceid.args.save_faces:
        faceid.makedirs(os.path.join(data_dir, faceid.args.save_faces))

    start_time = time.time()
    db.connect(os.path.join(data_dir, dbfile), args.debug)
    num_files = 0
    num_images = 0
    num_faces = 0
    results = []
    for file in files:
        file_images, file_faces, res = faceid.process_file(data_dir, file, source)
        num_files += 1
        num_images += file_images
        num_faces += file_faces
        results += res
    db.close()
    elapsed = time.time() - start_time

    res = {'num_files': num_files, 'num_images': num_images, 'num_faces': num_faces, 'elapsed': elapsed, 'images_per_s': num_images / elapsed, 'files': results}
    return jsonify(res)

@app.route("/compute_similarities", methods=['POST'])
def compute_similarities():
    params = request.get_json()

    start_time = time.time()
    db.connect(os.path.join(params['data_dir'], params['dbfile']), args.debug)
    num_faces, num_similarities, num_clusters = faceid.compute_similarities(
            float(params.get('similarity_threshold', args.similarity_threshold)), 
            float(params.get('identity_threshold', args.identity_threshold))
    )
    db.close()
    elapsed = time.time() - start_time

    res = {'num_similarities': num_similarities, 'num_clusters': num_clusters, 'num_faces': num_faces, 'elapsed': elapsed}
    return jsonify(res)

@app.route("/get_clusters", methods=['POST'])
def get_clusters():
    params = request.get_json()

    db.connect(os.path.join(params['data_dir'], params['dbfile']), args.debug)
    res = db.get_clusters(
            float(params.get('confidence_threshold', args.confidence_threshold)), 
            params.get('with_gps', False), 
            int(params.get('limit', 5))
    )
    db.close()

    return jsonify(res)

@app.route("/get_cluster_faces", methods=['POST'])
def get_cluster_faces():
    params = request.get_json()

    db.connect(os.path.join(params['data_dir'], params['dbfile']), args.debug)
    res = db.get_cluster_faces(
            int(params['cluster_num']),
            params.get('with_gps', False), 
            int(params.get('limit', 5))
    )
    db.close()

    return jsonify(res)

@app.route("/get_similar_faces", methods=['POST'])
def get_similar_faces():
    params = request.get_json()

    db.connect(os.path.join(params['data_dir'], params['dbfile']), args.debug)
    res = db.get_similar_faces(
            int(params['face_id']),
            int(params.get('limit', 5)),
            float(params.get('similarity_threshold', args.similarity_threshold))
    )
    db.close()

    return jsonify(res)

@app.route("/get_selfies", methods=['POST'])
def get_selfies():
    params = request.get_json()

    db.connect(os.path.join(params['data_dir'], params['dbfile']), args.debug)
    res = db.get_selfies(
            int(params.get('limit', 5))
    )
    db.close()

    return jsonify(res)

@app.route("/get_criminals", methods=['POST'])
def get_criminals():
    params = request.get_json()

    db.connect(os.path.join(params['data_dir'], params['dbfile']), args.debug)
    res = db.get_criminals(
            int(params['face_id']), 
            #int(params['cluster_num']),
            int(params.get('limit', 5)),
            float(params.get('similarity_threshold', args.similarity_threshold))
    )
    db.close()

    return jsonify(res)

if __name__ == '__main__':
    app.run(args.host, args.port, debug=args.debug_server)
