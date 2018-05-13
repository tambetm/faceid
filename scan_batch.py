from __future__ import print_function
import dlib
import cv2
import magic
import exifread
import numpy as np
import sqlite3
import json
import time
import os
import argparse

def create_tables():
    conn.execute("""CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY NOT NULL,
    type TEXT CHECK(type IN ('image', 'video', 'pbimage', 'pbvideo', 'google', 'crime')) NOT NULL,
    filepath TEXT NOT NULL,
    image_width INTEGER NOT NULL, 
    image_height INTEGER NOT NULL, 
    resized_filepath TEXT, 
    resized_width INTEGER NOT NULL, 
    resized_height INTEGER NOT NULL, 
    frame_num INTEGER,
    orientation TEXT CHECK(orientation IN ('portrait', 'landscape')),
    num_faces INTEGER NOT NULL,
    exif_data TEXT,
    UNIQUE(filepath, frame_num)
)""")

    conn.execute("""CREATE TABLE IF NOT EXISTS faces (
    face_id INTEGER PRIMARY KEY NOT NULL,
    image_id INTEGER NOT NULL,
    face_num INTEGER NOT NULL,
    left REAL NOT NULL,
    top REAL NOT NULL,
    right REAL NOT NULL,
    bottom REAL NOT NULL,
    width REAL NOT NULL,
    height REAL NOT NULL,
    landmarks TEXT NOT NULL, 
    descriptor TEXT NOT NULL,
    FOREIGN KEY (image_id) REFERENCES images(image_id)
    UNIQUE(image_id, face_num)
)""")

    conn.execute("""CREATE TABLE IF NOT EXISTS similarities (
    face1_id INTEGER NOT NULL,
    face2_id INTEGER NOT NULL,
    distance REAL NOT NULL,
    FOREIGN KEY (face1_id) REFERENCES faces(face_id)
    FOREIGN KEY (face2_id) REFERENCES faces(face_id)
    UNIQUE(face1_id, face2_id)
)""")

def insert_image(row):
    c = conn.cursor()
    c.execute("""INSERT INTO images 
    (type, filepath, image_width, image_height, resized_filepath, resized_width, resized_height, frame_num, exif_data, num_faces) 
    VALUES (?,?,?,?,?,?,?,?,?,?)""", row)
    return c.lastrowid

def insert_face(row):
    c = conn.cursor()
    c.execute("""INSERT INTO faces 
    (image_id, face_num, left, top, right, bottom, width, height, landmarks, descriptor) 
    VALUES (?,?,?,?,?,?,?,?,?,?)""", row)
    return c.lastrowid

def delete_similarities():
    conn.execute("DELETE FROM similarities")

def insert_similarity(row):
    c = conn.cursor()
    c.execute("""INSERT INTO similarities 
    (face1_id, face2_id, distance) 
    VALUES (?,?,?)""", row)
    return c.lastrowid

def get_all_descriptors():
    c = conn.cursor()
    c.execute("SELECT face_id, descriptor FROM faces")
    return c.fetchall()

def file_exists(filepath):
    c = conn.cursor()
    c.execute("SELECT 1 FROM images WHERE filepath = ?", (filepath,))
    return c.fetchone() is not None

def resize_image(img, size):
    fx = float(size) / img.shape[1]
    fy = float(size) / img.shape[0]
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

    faces_queue = detector(grays_queue, args.upscale, batch_size=len(grays_queue))

    for faces, img, gray, data in zip(faces_queue, images_queue, grays_queue, data_queue):
        image_id = insert_image(data + [len(faces)])

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
            
            face_id = insert_face([image_id, i, face_left, face_top, face_right, face_bottom, face_width, face_height, json.dumps(landmarks), json.dumps(descriptor)])

            if args.save_resized:
                facepath = os.path.join(args.save_resized, "face_%02d.jpg" % face_id)
                cv2.imwrite(facepath, img[rect.top():rect.bottom(),rect.left():rect.right()])

        num_images += 1
        num_faces += len(faces)

    conn.commit()

    images_queue.clear()    
    grays_queue.clear()    
    data_queue.clear()    

    elapsed = time.time() - start_time
    print("\rFiles: %d, Images: %d, Faces: %d, Elapsed: %.2fs, Average per image: %.3fs" % (num_files, num_images, num_faces, elapsed, elapsed / num_images), end='')

def process_image_file(filepath):
    #print('Image:', filepath)
    if file_exists(filepath):
        return
    img = cv2.imread(filepath)
    if img is None:
        return
    with open(filepath, 'rb') as f:
        tags = exifread.process_file(f, details=False)
        tags = {k:str(v) for k, v in tags.items()}
    process_image(filepath, img, get_image_type('image'), exif_data=json.dumps(tags))

def process_video_file(filepath):
    #print('Video:', filepath)
    if file_exists(filepath):
        return
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
        ret = cap.grab()
        if not ret:
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
    face_descriptors = get_all_descriptors()
    #print("get_all_descriptors():", t)

    print("Faces: %d" % len(face_descriptors), end='')
    if len(face_descriptors) == 0:
        print()
        return

    descriptors = np.array([json.loads(f[1]) for f in face_descriptors])
    #print("convert to array:", t)
    #dists = np.sqrt(np.sum(np.square(descriptors[np.newaxis] - descriptors[:, np.newaxis]), axis=-1))
    sumsquares = np.sum(np.square(descriptors), axis=-1)
    dists = np.sqrt(np.maximum(sumsquares[np.newaxis] + sumsquares[:, np.newaxis] - 2 * np.dot(descriptors, descriptors.T), 0))
    #assert np.allclose(dists, dists2, atol=1e-7)
    #print("calculate dists:", t)
    delete_similarities()
    #print("delete similarities:", t)
    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            if i != j and dists[i, j] < args.similarity_threshold:
                insert_similarity([face_descriptors[i][0], face_descriptors[j][0], dists[i, j]])
    #print("save similarities:", t)
    conn.commit()
    #print("commit:", t)
    print(", Time: %.2fs" % t.total())

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
parser.add_argument("--similarity_threshold", type=float, default=0.5)
parser.add_argument("--video_max_images", type=int, default=10)
parser.add_argument("--type", choices=['default', 'photobooth', 'google', 'crime'], default='default')
args = parser.parse_args()

detector = dlib.cnn_face_detection_model_v1(args.cnn_model_path)
predictor = dlib.shape_predictor(args.predictor_path)
facerec = dlib.face_recognition_model_v1(args.face_rec_model_path)

if args.save_resized:
    try: 
        os.makedirs(args.save_resized)
    except OSError:
        if not os.path.isdir(args.save_resized):
            raise

conn = sqlite3.connect(args.db)
create_tables()

print("Processing files...")
start_time = time.time()
num_files = 0
num_images = 0
num_faces = 0
for dirpath, dirnames, filenames in os.walk(args.dir):
    for filename in filenames:
        filepath = os.path.join(dirpath, filename)        
        mime_type = magic.from_file(filepath, mime=True)
        if mime_type.startswith('image/'):
            process_image_file(filepath)
        elif mime_type.startswith('video/'):
            process_video_file(filepath)
        num_files += 1
# process remaining images in the queue
process_queue()
print()

print("Calculating similarities...")
compute_similarities()

print("Done")
