from __future__ import print_function
import dlib
import cv2
import magic
import exifread
import numpy as np
import facedb as db
import sqlite3
import json
import time
import os
import argparse

def resize_image(img, max_size):
    fx = float(max_size) / img.shape[1]
    fy = float(max_size) / img.shape[0]
    f = min(fx, fy, 1)
    return cv2.resize(img, (0, 0), fx=f, fy=f)

def get_image_type(image_type):
    if args.type == 'default':
        return image_type
    elif args.type == 'photobooth':
        return 'pb' + image_type
    else:
        return args.type

def process_image(filepath, img, image_type, frame_num=None, exif_data=None):
    image_height, image_width, _ = img.shape
    resizepath, resized_height, resized_width = None, image_height, image_width

    if args.resize:
        img = resize_image(img, args.resize)
        if args.save_resized:
            filename = os.path.basename(filepath)
            resizepath = os.path.join(args.save_resized, filename)
            basepath, ext = os.path.splitext(resizepath)
            if ext == '' or frame_num is not None:
                resizepath = basepath
                if frame_num is not None:
                    resizepath += "_%04d" % frame_num
                resizepath += '.jpg'
            cv2.imwrite(resizepath, img)
            resized_height, resized_width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, args.upscale)

    image_id = db.insert_image([image_type, filepath, image_width, image_height, resizepath, resized_width, resized_height, frame_num, exif_data, len(faces)])

    poses = dlib.full_object_detections()
    rects = []
    for rect in faces:
        if args.detector == 'cnn':
            rect = rect.rect
        pose = predictor(gray, rect)
        poses.append(pose)
        rects.append(rect)

    # do batched computation of face descriptors
    img_rgb = img[...,::-1] # BGR to RGB
    descriptors = facerec.compute_face_descriptor(img_rgb, poses, args.jitter)

    for i, (rect, pose, descriptor) in enumerate(zip(rects, poses, descriptors)):
        face_left = float(rect.left()) / resized_width
        face_top = float(rect.top()) / resized_height
        face_right = float(rect.right()) / resized_width
        face_bottom = float(rect.bottom()) / resized_height
        face_width = face_right - face_left
        face_height = face_bottom - face_top
        landmarks = [[float(p.x) / resized_width, float(p.y) / resized_height] for p in pose.parts()]
        descriptor = list(descriptor)
        
        face_id = db.insert_face([image_id, i, face_left, face_top, face_right, face_bottom, face_width, face_height, json.dumps(landmarks), json.dumps(descriptor)])

        if args.save_resized:
            facepath = os.path.join(args.save_resized, "face_%02d.jpg" % face_id)
            cv2.imwrite(facepath, img[rect.top():rect.bottom(),rect.left():rect.right()])

    db.commit()

    global num_images
    global num_faces
    num_images += 1
    num_faces += len(faces)
    elapsed = time.time() - start_time
    print("\rFiles: %d, Images: %d, Faces: %d, Elapsed: %.2fs, Average per image: %.3fs" % (num_files, num_images, num_faces, elapsed, elapsed / num_images), end='')

def process_image_file(filepath):
    #print('Image:', filepath)
    img = cv2.imread(filepath)
    if img is None:
        return
    with open(filepath, 'rb') as f:
        tags = exifread.process_file(f, details=False)
        tags = {k:str(v) for k, v in tags.items()}
    process_image(filepath, img, get_image_type('image'), exif_data=json.dumps(tags))

def process_video_file(filepath):
    #print('Video:', filepath)
    cap = cv2.VideoCapture(filepath)
    #print()
    #print("FPS:", cap.get(cv2.CAP_PROP_FPS), "frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(round(cap.get(cv2.CAP_PROP_FPS)), round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / args.video_max_images))
    #frame_interval = round(cap.get(cv2.CAP_PROP_FPS))
    frame_num = 0
    used_images = 0
    while cap.isOpened() and used_images < args.video_max_images:
        #print("msec:", cap.get(cv2.CAP_PROP_POS_MSEC), "frame:", cap.get(cv2.CAP_PROP_POS_FRAMES), )
        # just grab the frame, do not decode
        if not cap.grab():
            break
        # process one frame per second
        if frame_num % frame_interval == 0:
            # decode the grabbed frame
            ret, img = cap.retrieve()
            assert ret
            process_image(filepath, img, get_image_type('video'), frame_num=frame_num)
            used_images += 1
        frame_num += 1
    cap.release()

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

def compute_similarities():
    t = Timer()
    face_descriptors = db.get_all_descriptors()
    #print("get_all_descriptors():", t)

    print("Faces: %d" % len(face_descriptors), end='')
    if len(face_descriptors) == 0:
        print()
        return

    descriptors = np.array([json.loads(f[1]) for f in face_descriptors])
    #print("convert to array:", t)
    sumsquares = np.sum(np.square(descriptors), axis=-1)
    dists = np.sqrt(np.maximum(sumsquares[np.newaxis] + sumsquares[:, np.newaxis] - 2 * np.dot(descriptors, descriptors.T), 0))
    #print("calculate dists:", t)
    db.delete_similarities()
    #print("delete similarities:", t)
    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            if i != j and dists[i, j] < args.similarity_threshold:
                db.insert_similarity([face_descriptors[i][0], face_descriptors[j][0], dists[i, j]])
    #print("save similarities:", t)
    db.commit()
    #print("commit:", t)
    print(", Time: %.2fs" % t.total())

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("db")
parser.add_argument("--detector", choices=['hog', 'cnn'], default='hog')
parser.add_argument("--upscale", type=int, default=0)
parser.add_argument("--jitter", type=int, default=0)
parser.add_argument("--predictor_path", default='shape_predictor_68_face_landmarks.dat')
parser.add_argument("--face_rec_model_path", default='dlib_face_recognition_resnet_model_v1.dat')
parser.add_argument("--cnn_model_path", default='mmod_human_face_detector.dat')
parser.add_argument("--resize", type=int, default=1024)
parser.add_argument("--save_resized")
parser.add_argument("--similarity_threshold", type=float, default=0.5)
parser.add_argument("--video_max_images", type=int, default=10)
parser.add_argument("--type", choices=['default', 'photobooth', 'google', 'crime'], default='default')
args = parser.parse_args()

if args.detector == 'hog':
    detector = dlib.get_frontal_face_detector()
elif args.detector == 'cnn':
    detector = dlib.cnn_face_detection_model_v1(args.cnn_model_path)
else:
    assert False, "Unknown detector " + args.detector
predictor = dlib.shape_predictor(args.predictor_path)
facerec = dlib.face_recognition_model_v1(args.face_rec_model_path)

if args.save_resized:
    try: 
        os.makedirs(args.save_resized)
    except OSError:
        if not os.path.isdir(args.save_resized):
            raise

db.connect(args.db)

print("Processing files...")
start_time = time.time()
num_files = 0
num_images = 0
num_faces = 0
for dirpath, dirnames, filenames in os.walk(args.dir):
    for filename in filenames:
        filepath = os.path.join(dirpath, filename)        
        if db.file_exists(filepath):
            continue
        num_files += 1
        mime_type = magic.from_file(filepath, mime=True)
        if mime_type.startswith('image/'):
            process_image_file(filepath)
        elif mime_type.startswith('video/'):
            process_video_file(filepath)
print()

print("Calculating similarities...")
compute_similarities()

print("Done")
