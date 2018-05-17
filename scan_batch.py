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
import argparse

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

def get_image_type(image_type):
    if args.type == 'default':
        return image_type
    elif args.type == 'photobooth':
        return 'pb' + image_type
    else:
        return args.type

images_queue = []
grays_queue = []
data_queue = []

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

    global images_queue
    global grays_queue
    global data_queue

    # if image has different resolution, then process queue
    if len(data_queue) > 0 and (data_queue[0][5] != resized_width or data_queue[0][6] != resized_height):
        process_queue()

    images_queue.append(img)
    grays_queue.append(gray)
    data_queue.append([image_type, filepath, image_width, image_height, resizepath, resized_width, resized_height, frame_num, exif_data])

    # if batch full then process queue
    if len(images_queue) == args.batch_size:
        process_queue()

def process_queue():
    global images_queue
    global grays_queue
    global data_queue
    global num_images
    global num_faces
    global num_files

    faces_queue = detector(grays_queue, args.upscale, batch_size=len(grays_queue))

    for faces, img, gray, data in zip(faces_queue, images_queue, grays_queue, data_queue):
        image_id = db.insert_image(data + [len(faces)])

        poses = dlib.full_object_detections()
        rects = []
        for face in faces:
            pose = predictor(gray, face.rect)
            poses.append(pose)
            rects.append(face.rect)

        # do batched computation of face descriptors
        img_rgb = img[...,::-1] # BGR to RGB
        descriptors = facerec.compute_face_descriptor(img_rgb, poses, args.jitter)

        resized_width = data[5]
        resized_height = data[6]
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

            if args.save_faces:
                facepath = os.path.join(args.save_faces, "face_%02d.jpg" % face_id)
                cv2.imwrite(facepath, img[rect.top():rect.bottom(),rect.left():rect.right()])

        num_images += 1
        num_faces += len(faces)

    db.commit()

    images_queue.clear()
    grays_queue.clear()
    data_queue.clear()

    elapsed = time.time() - start_time
    print("\rFiles: %d, Images: %d, Faces: %d, Elapsed: %.2fs, Images/s: %.1fs" % (num_files, num_images, num_faces, elapsed, num_images / elapsed), end='')

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
        # just grab the frame, do not decode
        if not cap.grab():
            break
        # process one frame after interval
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
    all_descriptors = db.get_all_descriptors()
    #print("get_all_descriptors():", t)
    print("Faces: %d" % len(all_descriptors), end='')
    if len(all_descriptors) < 2:
        print()
        return

    X = Y = np.array([json.loads(f[1]) for f in all_descriptors])
    #print("convert to array:", t)
    X2 = Y2 = np.sum(np.square(X), axis=-1)
    dists = np.sqrt(np.maximum(X2[:, np.newaxis] + Y2[np.newaxis] - 2 * np.dot(X, Y.T), 0))
    #print("calculate dists:", t)

    db.delete_similarities()
    #print("delete similarities:", t)
    num_similarities = 0
    for i, j in zip(*np.where(dists < args.similarity_threshold)):
        if i != j:
            db.insert_similarity([all_descriptors[i][0], all_descriptors[j][0], dists[i, j]])
            num_similarities += 1
    #print("save similarities:", t)
    db.commit()
    #print("commit:", t)
    print(", Similarities: %d, Time: %.2fs" % (num_similarities, t.total()))

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("db")
parser.add_argument("--upscale", type=int, default=0)
parser.add_argument("--jitter", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--predictor_path", default='shape_predictor_68_face_landmarks.dat')
parser.add_argument("--face_rec_model_path", default='dlib_face_recognition_resnet_model_v1.dat')
parser.add_argument("--cnn_model_path", default='mmod_human_face_detector.dat')
parser.add_argument("--resize", type=int, default=1024)
parser.add_argument("--save_resized")
parser.add_argument("--save_faces")
parser.add_argument("--similarity_threshold", type=float, default=0.5)
parser.add_argument("--video_max_images", type=int, default=10)
parser.add_argument("--type", choices=['default', 'photobooth', 'google', 'crime'], default='default')
args = parser.parse_args()

detector = dlib.cnn_face_detection_model_v1(args.cnn_model_path)
predictor = dlib.shape_predictor(args.predictor_path)
facerec = dlib.face_recognition_model_v1(args.face_rec_model_path)

if args.save_resized:
    makedirs(args.save_resized)

if args.save_faces:
    makedirs(args.save_faces)

db.connect(args.db)

print("Processing files...")
start_time = time.time()
num_files = 0
num_images = 0
num_faces = 0
for dirpath, dirnames, filenames in os.walk(args.dir):
    for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        num_files += 1
        if db.file_exists(filepath):
            continue
        mime_type = magic.from_file(filepath, mime=True)
        if mime_type.startswith('image/'):
            process_image_file(filepath)
        elif mime_type.startswith('video/'):
            process_video_file(filepath)
# process remaining images in the queue
process_queue()
print()

print("Calculating similarities...")
compute_similarities()

print("Done")
