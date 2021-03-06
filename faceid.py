from __future__ import print_function
import dlib
import cv2
import magic
import exifread
import numpy as np
import facedb as db
import json
import time
import os
import shutil
import argparse

args = None
detector = None
predictor = None
facerec = None

def argparser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--detector", choices=['hog', 'cnn'], default='cnn')
    parser.add_argument("--upscale", type=int, default=0)
    parser.add_argument("--jitter", type=int, default=10)
    parser.add_argument("--predictor_path", default='shape_predictor_68_face_landmarks.dat')
    parser.add_argument("--face_rec_model_path", default='dlib_face_recognition_resnet_model_v1.dat')
    parser.add_argument("--cnn_model_path", default='mmod_human_face_detector.dat')
    parser.add_argument("--resize", type=int, default=1024)
    parser.add_argument("--save_resized", default="resized")
    parser.add_argument("--save_faces", default="faces")
    parser.add_argument("--draw_faces", action="store_true", default=True)
    parser.add_argument("--save_clusters", default="clusters")
    parser.add_argument("--source", choices=['phone', 'photobooth', 'google', 'interpol'], default='phone')
    parser.add_argument("--identity_threshold", type=float, default=0.4)
    parser.add_argument("--similarity_threshold", type=float, default=0.6)
    parser.add_argument("--confidence_threshold", type=float, default=0.95)
    parser.add_argument("--video_max_images", type=int, default=10)
    return parser

def init(_args):
    global args
    global detector
    global predictor
    global facerec

    args = _args

    if args.detector == 'hog':
        detector = dlib.get_frontal_face_detector()
    elif args.detector == 'cnn':
        detector = dlib.cnn_face_detection_model_v1(args.cnn_model_path)
    else:
        assert False, "Unknown detector " + args.detector
    predictor = dlib.shape_predictor(args.predictor_path)
    facerec = dlib.face_recognition_model_v1(args.face_rec_model_path)

def makedirs(dir):
    try: 
        os.makedirs(dir)
    except OSError:
        if not os.path.isdir(dir):
            raise

def resize_image(img, max_size):
    fx = float(max_size) / img.shape[1]
    fy = float(max_size) / img.shape[0]
    f = min(fx, fy, 1)
    return cv2.resize(img, (0, 0), fx=f, fy=f)

def process_image(data_dir, relpath, img, image_type, image_source, frame_num=None, exif_data=None, 
        gps_lat=None, gps_lon=None, camera_side=None, rotate=0, timestamp=None, **kwargs):
    #if int(rotate) != 0:
    #    assert int(rotate) in [0, 90, 180, 270]
    #    img = cv2.rotate(img, int(rotate) // 90 - 1)
    #    print("Rotating", rotate)

    image_height, image_width, _ = img.shape
    resizepath, resized_height, resized_width = None, image_height, image_width

    if args.resize:
        img = resize_image(img, args.resize)
        resized_height, resized_width, _ = img.shape
        if args.save_resized:
            filename = os.path.basename(relpath)
            resizepath = os.path.join(args.save_resized, filename)
            basepath, ext = os.path.splitext(resizepath)
            if ext == '' or frame_num is not None:
                resizepath = basepath
                if frame_num is not None:
                    resizepath += "_%04d" % frame_num
                resizepath += '.jpg'

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, args.upscale)

    image_id = db.insert_image([image_type, image_source, relpath, image_width, image_height, 
        resizepath, resized_width, resized_height, frame_num, exif_data, len(faces),
        gps_lat, gps_lon, camera_side, rotate, timestamp])

    poses = dlib.full_object_detections()
    rects = []
    confs = []
    for rect in faces:
        if args.detector == 'cnn':
            confs.append(rect.confidence)
            rect = rect.rect
        else:
            confs.append(1.)
        pose = predictor(gray, rect)
        poses.append(pose)
        rects.append(rect)

    # do batched computation of face descriptors
    img_rgb = img[...,::-1] # BGR to RGB
    descriptors = facerec.compute_face_descriptor(img_rgb, poses, args.jitter)

    faceres = []
    for i, (rect, conf, pose, descriptor) in enumerate(zip(rects, confs, poses, descriptors)):
        face_left = float(rect.left()) / resized_width
        face_top = float(rect.top()) / resized_height
        face_right = float(rect.right()) / resized_width
        face_bottom = float(rect.bottom()) / resized_height
        face_width = face_right - face_left
        face_height = face_bottom - face_top
        landmarks = [[float(p.x) / resized_width, float(p.y) / resized_height] for p in pose.parts()]
        descriptor = list(descriptor)
        
        face_id = db.insert_face([image_id, i, face_left, face_top, face_right, face_bottom, face_width, face_height, conf, json.dumps(landmarks), json.dumps(descriptor)])

        # draw faces on resized images
        if args.draw_faces:
            cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 255), 2)
            for p in pose.parts():
                cv2.circle(img, (p.x, p.y), 1, color=(0, 255, 255), thickness=-1)

        if args.save_faces:
            facepath = os.path.join(data_dir, args.save_faces, "face_%05d.jpg" % face_id)
            cv2.imwrite(facepath, img[max(rect.top(), 0):rect.bottom(), max(rect.left(), 0):rect.right()])

        faceres.append({'face_id': face_id, 'face_num': i, 'left': face_left, 'top': face_top, 'right': face_right, 'bottom': face_bottom, 'width': face_width, 'height': face_height,
            'confidence': conf, 'landmarks': landmarks, 'pose_coef': (1 - face_bottom) / face_height})

    db.commit()

    # save resized image after drawing faces
    if args.resize and args.save_resized:
        cv2.imwrite(os.path.join(data_dir, resizepath), img)

    res = {'relpath': relpath, 'frame_num': frame_num, 'resizepath': resizepath, 'image_id': image_id, 'image_type': image_type, 'num_faces': len(faces), 'faces': faceres}
    return len(faces), res

def process_image_file(data_dir, relpath, source, **kwargs):
    filepath = os.path.join(data_dir, relpath)
    #print('Image:', filepath)
    img = cv2.imread(filepath)
    if img is None:
        return 0, 0, []
    with open(filepath, 'rb') as f:
        tags = exifread.process_file(f, details=False)
        tags = {k:str(v) for k, v in tags.items()}
    num_faces, res = process_image(data_dir, relpath, img, 'image', source, exif_data=json.dumps(tags), **kwargs)
    return 1, num_faces, [res]

def process_video_file(data_dir, relpath, source, **kwargs):
    filepath = os.path.join(data_dir, relpath)
    #print('Video:', filepath)
    cap = cv2.VideoCapture(filepath)
    #print()
    #print("FPS:", cap.get(cv2.CAP_PROP_FPS), "frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(round(cap.get(cv2.CAP_PROP_FPS)), round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / args.video_max_images))
    #frame_interval = round(cap.get(cv2.CAP_PROP_FPS))
    frame_num = 0
    num_images = 0
    num_faces = 0
    results = []
    while cap.isOpened() and num_images < args.video_max_images:
        #print("msec:", cap.get(cv2.CAP_PROP_POS_MSEC), "frame:", cap.get(cv2.CAP_PROP_POS_FRAMES), )
        # just grab the frame, do not decode
        if not cap.grab():
            break
        # process one frame per second
        if frame_num % frame_interval == 0:
            # decode the grabbed frame
            ret, img = cap.retrieve()
            assert ret
            image_faces, res = process_image(data_dir, relpath, img, 'video', source, frame_num=frame_num, **kwargs)
            num_images += 1
            num_faces += image_faces
            results.append(res)
        frame_num += 1
    cap.release()
    return num_images, num_faces, results

def process_file(data_dir, relpath, source, **kwargs):
    if db.file_exists(relpath):
        return 0, 0, []
    filepath = os.path.join(data_dir, relpath)
    mime_type = magic.from_file(filepath, mime=True)
    if mime_type.startswith('image/'):
        return process_image_file(data_dir, relpath, source, **kwargs)
    elif mime_type.startswith('video/'):
        return process_video_file(data_dir, relpath, source, **kwargs)
    else:
        return 0, 0, []

class Timer(object):
    def __init__(self):
        self.start = time.time()
        self.time = time.time()
    def t(self):
        elapsed = time.time() - self.time
        self.time = time.time()
        return elapsed
    def total(self):
        return time.time() - self.start
    def __str__(self):
        return str(self.t())

def compute_similarities(data_dir, similarity_threshold=0.6, identity_threshold=0.4, criminal_fraction=0.1, **kwargs):
    t = Timer()
    all_descriptors = db.get_all_descriptors()
    descriptors = [json.loads(f[1]) for f in all_descriptors]
    face_ids = [f[0] for f in all_descriptors]
    num_faces = len(all_descriptors)
    #print("get_all_descriptors():", t)
    #print("Faces: %d" % len(all_descriptors), end='')
    if num_faces < 2:
        #print()
        return num_faces, 0, 0

    X = Y = np.array(descriptors)
    #print("convert to array:", t)
    X2 = Y2 = np.sum(np.square(X), axis=-1)
    dists = np.sqrt(np.maximum(X2[:, np.newaxis] + Y2[np.newaxis] - 2 * np.dot(X, Y.T), 0))
    #print("calculate dists:", t)

    db.delete_similarities()
    #print("delete similarities:", t)
    num_similarities = 0
    for i, j in zip(*np.where(dists < float(similarity_threshold))):
        if i != j:
            db.insert_similarity([face_ids[i], face_ids[j], dists[i, j]])
            num_similarities += 1
    #print("save similarities:", t)

    # cluster faces and update labels
    descriptors_dlib = [dlib.vector(d) for d in descriptors]
    clusters = dlib.chinese_whispers_clustering(descriptors_dlib, float(identity_threshold))
    db.update_labels(zip(clusters, face_ids))
    num_clusters = len(set(clusters))

    if args.save_clusters:
        for cluster_num, face_id in zip(clusters, face_ids):
            facefile = os.path.realpath(os.path.join(data_dir, args.save_faces, "face_%05d.jpg" % face_id))
            clusterdir = os.path.join(data_dir, args.save_clusters, str(cluster_num))
            makedirs(clusterdir)
            os.symlink(facefile, os.path.join(clusterdir, 'tmpfile'))
            os.rename(os.path.join(clusterdir, 'tmpfile'), os.path.join(clusterdir, "face_%05d.jpg" % face_id))

    # remove clusters with more than given amount of criminals
    criminal_clusters = db.get_clusters_with_criminals(criminal_fraction)
    for cluster in criminal_clusters:
        db.remove_cluster(cluster['cluster_num'])

    db.commit()
    #print("commit:", t)
    #print(", Similarities: %d, Time: %.2fs" % (num_similarities, t.total()))
    return num_faces, num_similarities, num_clusters

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[argparser()])
    parser.add_argument("dir")
    parser.add_argument("db")
    args = parser.parse_args()

    init(args)

    if args.save_resized:
        resized_dir = os.path.join(args.dir, args.save_resized)
        shutil.rmtree(resized_dir)
        makedirs(resized_dir)

    if args.save_faces:
        faces_dir = os.path.join(args.dir, args.save_faces)
        shutil.rmtree(faces_dir)
        makedirs(faces_dir)

    if args.save_clusters:
        clusters_dir = os.path.join(args.dir, args.save_clusters)
        shutil.rmtree(clusters_dir)
        makedirs(clusters_dir)

    db.connect(args.db)

    print("Processing files...")
    start_time = time.time()
    num_files = 0
    num_images = 0
    num_faces = 0
    for dirpath, dirnames, filenames in os.walk(args.dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            relpath = os.path.relpath(filepath, args.dir)
            file_images, file_faces, results = process_file(args.dir, relpath, args.source)
            num_files += 1
            num_images += file_images
            num_faces += file_faces
            elapsed = time.time() - start_time
            print("\rFiles: %d, Images: %d, Faces: %d, Elapsed: %.2fs, Images/s: %.1fs" % (num_files, num_images, num_faces, elapsed, num_images / elapsed), end='')
    print()

    print("Calculating similarities...")
    compute_similarities(args.similarity_threshold, args.identity_threshold)

    print("Done")
