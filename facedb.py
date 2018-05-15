import sqlite3

conn = None

def connect(db):
    global conn
    conn = sqlite3.connect(db)
    create_tables()

def commit():
    conn.commit()

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

def insert_similarities(rows):
    c = conn.cursor()
    c.executemany("""INSERT INTO similarities 
    (face1_id, face2_id, distance) 
    VALUES (?,?,?)""", rows)

def get_all_descriptors():
    c = conn.cursor()
    c.execute("SELECT face_id, descriptor FROM faces")
    return c.fetchall()

def file_exists(filepath):
    c = conn.cursor()
    c.execute("SELECT 1 FROM images WHERE filepath = ?", (filepath,))
    return c.fetchone() is not None

def get_common_faces(with_gps=False, limit=5, similarity_threshold=0.35):
    c = conn.cursor()
    c.execute("""SELECT s.face1_id, count(1) 
    FROM similarities s 
    JOIN faces f1 ON s.face1_id = f1.face_id 
    JOIN faces f2 ON s.face2_id = f2.face_id 
    JOIN images i1 ON f1.image_id = i1.image_id 
    JOIN images i2 ON f2.image_id = i2.image_id 
    WHERE s.distance < ? 
        AND i1.type IN ('image', 'video')
        AND i2.type IN ('image', 'video')
    GROUP BY s.face1_id
    ORDER BY count(1) DESC
    LIMIT ?""", (similarity_threshold, limit,))
    return c.fetchall()

def get_similar_faces(face_id, limit=5, similarity_threshold=0.35):
    c = conn.cursor()
    c.execute("""SELECT f.*, i.*
    FROM similarities s 
    JOIN faces f ON s.face2_id = f.face_id 
    JOIN images i ON f.image_id = i.image_id 
    WHERE s.face1_id = ? AND s.distance < ? 
    ORDER BY s.distance
    LIMIT ?""", (face_id, similarity_threshold, limit,))
    return c.fetchall()

def get_my_face():
    c = conn.cursor()
    c.execute("""SELECT f.*, i.*
    FROM images i
    JOIN faces f ON f.image_id = i.image_id 
    WHERE i.type = 'pbimage'""")
    return c.fetchone()

def get_selfies(limit=5, similarity_threshold=0.35):
    me = get_my_face()
    assert me is not None
    c = conn.cursor()
    c.execute("""SELECT f2.*, i2.*
    FROM similarities s 
    JOIN faces f ON s.face2_id = f.face_id 
    JOIN images i ON f.image_id = i.image_id 
    WHERE s.face1_id = ? AND s.distance < ? 
        AND i.type IN ('image', 'video')
    ORDER BY s.distance
    LIMIT ?""", (me[0], similarity_threshold, limit,))
    return c.fetchall()

def get_selfies(limit=5, similarity_threshold=0.35):
    me = get_my_face()
    assert me is not None
    c = conn.cursor()
    c.execute("""SELECT f2.*, i2.*
    FROM similarities s
    JOIN faces f1 ON s.face1_id = f1.face_id 
    JOIN faces f2 ON s.face2_id = f2.face_id 
    JOIN images i1 ON f1.image_id = i1.image_id 
    JOIN images i2 ON f2.image_id = i2.image_id 
    WHERE s.distance < ? 
        AND i1.type = 'pbimage'
        AND i2.type IN ('image', 'video')
    ORDER BY s.distance
    LIMIT ?""", (me[0], similarity_threshold, limit,))
    return c.fetchall()

def get_criminals(face_id, limit=5, similarity_threshold=0.35):
    c = conn.cursor()
    c.execute("""SELECT f.*, i.*
    FROM similarities s
    JOIN faces f ON s.face2_id = f.face_id 
    JOIN images i ON f.image_id = i.image_id 
    WHERE s.face1_id = ? 
        AND s.distance < ?
        AND i.type = 'crime'
    ORDER BY s.distance
    LIMIT ?""", (face_id, similarity_threshold, limit,))
    return c.fetchall()
