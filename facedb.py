import sqlite3

conn = None

def connect(db, debug=False):
    global conn
    conn = sqlite3.connect(db)
    if debug:
        conn.set_trace_callback(print)
    create_tables()

def commit():
    conn.commit()

def close():
    conn.close()

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def create_tables():
    conn.executescript("""
CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY NOT NULL,
    type TEXT CHECK(type IN ('image', 'video')) NOT NULL,
    source TEXT CHECK(source IN ('phone', 'photobooth', 'google', 'interpol')) NOT NULL,
    filepath TEXT NOT NULL,
    image_width INTEGER NOT NULL, 
    image_height INTEGER NOT NULL, 
    resized_filepath TEXT, 
    resized_width INTEGER NOT NULL, 
    resized_height INTEGER NOT NULL, 
    frame_num INTEGER,
    num_faces INTEGER NOT NULL,
    exif_data TEXT,
    gps_lat REAL,
    gps_lon REAL,
    rotate INTEGER,
    timestamp TEXT,
    UNIQUE(filepath, frame_num)
);
CREATE INDEX IF NOT EXISTS images_source_type_idx ON images(source, type);
CREATE TABLE IF NOT EXISTS faces (
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
    cluster_num INTEGER,
    FOREIGN KEY (image_id) REFERENCES images(image_id)
    UNIQUE(image_id, face_num)
);
CREATE INDEX IF NOT EXISTS images_cluster_num_idx ON faces(cluster_num);
CREATE TABLE IF NOT EXISTS similarities (
    face1_id INTEGER NOT NULL,
    face2_id INTEGER NOT NULL,
    distance REAL NOT NULL,
    FOREIGN KEY (face1_id) REFERENCES faces(face_id)
    FOREIGN KEY (face2_id) REFERENCES faces(face_id)
    UNIQUE(face1_id, face2_id)
);
CREATE INDEX IF NOT EXISTS similarities_distance_idx ON similarities(distance);
""")

def insert_image(row):
    c = conn.cursor()
    c.execute("""INSERT INTO images 
    (type, source, filepath, image_width, image_height, resized_filepath, resized_width, resized_height, frame_num, exif_data, num_faces, gps_lat, gps_lon, rotate, timestamp) 
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", row)
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
    c.execute("INSERT INTO similarities (face1_id, face2_id, distance) VALUES (?,?,?)", row)
    return c.lastrowid

def insert_similarities(rows):
    c = conn.cursor()
    c.executemany("INSERT INTO similarities (face1_id, face2_id, distance) VALUES (?,?,?)", rows)

def get_all_descriptors():
    c = conn.cursor()
    c.execute("SELECT face_id, descriptor FROM faces")
    return c.fetchall()

def get_clusterable_descriptors():
    c = conn.cursor()
    c.execute("SELECT face_id, descriptor FROM faces f JOIN images i ON i.image_id = f.image_id WHERE i.source IN ('phone', 'photobooth')")
    return c.fetchall()

def update_labels(rows):
    c = conn.cursor()
    c.executemany("UPDATE faces SET cluster_num = ? WHERE face_id = ?", rows)

def file_exists(filepath):
    c = conn.cursor()
    c.execute("SELECT 1 FROM images WHERE filepath = ?", (filepath,))
    return c.fetchone() is not None

def get_clusters(with_gps=False, limit=5):
    conn.row_factory = dict_factory
    c = conn.cursor()
    conn.row_factory = None
    c.execute("""SELECT f.cluster_num, count(1) as count
    FROM faces f 
    JOIN (SELECT min(image_id), * images GROUP BY filepath)i ON i.image_id = f.image_id
    WHERE (NOT ? OR (i.gps_lat IS NOT NULL AND i.gps_lon IS NOT NULL))
        AND i.source = 'phone'
    GROUP BY f.cluster_num
    ORDER BY count(1) DESC
    LIMIT ?""", (with_gps, limit,))
    return c.fetchall()

def get_cluster_faces(cluster_num, with_gps=False, limit=5):
    conn.row_factory = dict_factory
    c = conn.cursor()
    conn.row_factory = None
    c.execute("""SELECT f.*, i.*
    FROM faces f 
    JOIN (SELECT min(image_id), * images GROUP BY filepath) i ON i.image_id = f.image_id
    WHERE f.cluster_num = ?
        AND (NOT ? OR (i.gps_lat IS NOT NULL AND i.gps_lon IS NOT NULL))
        AND i.source = 'phone'
    ORDER BY f.face_id
    LIMIT ?""", (cluster_num, with_gps, limit,))
    return c.fetchall()

def get_similar_faces(face_id, limit=5, similarity_threshold=0.35):
    conn.row_factory = dict_factory
    c = conn.cursor()
    conn.row_factory = None
    c.execute("""SELECT min(i.image_id) as image_id, f.*, i.*
    FROM similarities s 
    JOIN faces f ON s.face2_id = f.face_id 
    JOIN images i ON f.image_id = i.image_id 
    WHERE s.face1_id = ? AND s.distance < ?
    GROUP BY i.filepath
    ORDER BY s.distance
    LIMIT ?""", (face_id, similarity_threshold, limit,))
    return c.fetchall()

def get_my_face():
    c = conn.cursor()
    c.execute("""SELECT f.*, i.*
    FROM images i
    JOIN faces f ON f.image_id = i.image_id 
    WHERE i.type = 'image' AND i.source = 'photobooth'""")
    return c.fetchone()

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
        AND i1.type = 'image' AND i1.source = 'photobooth'
        AND i2.source = 'phone'
    ORDER BY s.distance
    LIMIT ?""", (similarity_threshold, limit,))
    return c.fetchall()

def get_selfies(limit=5):
    conn.row_factory = dict_factory
    c = conn.cursor()
    conn.row_factory = None
    c.execute("""SELECT f2.*, i2.*
    FROM faces f1 
    JOIN faces f2 ON f1.cluster_num = f2.cluster_num
    JOIN similarities s ON f1.face_id = s.face1_id AND f2.face_id = s.face2_id
    JOIN images i1 ON f1.image_id = i1.image_id 
    JOIN images i2 ON f2.image_id = i2.image_id 
    WHERE i1.type = 'image' AND i1.source = 'photobooth'
        AND i2.source = 'phone'
    ORDER BY s.distance
    LIMIT ?""", (limit,))
    return c.fetchall()

def get_criminals(face_id, limit=5, similarity_threshold=0.35):
    conn.row_factory = dict_factory
    c = conn.cursor()
    conn.row_factory = None
    c.execute("""SELECT f.*, i.*
    FROM similarities s
    JOIN faces f ON s.face2_id = f.face_id 
    JOIN images i ON f.image_id = i.image_id 
    WHERE s.face1_id = ? AND s.distance < ?
        AND i.source = 'interpol'
    ORDER BY s.distance
    LIMIT ?""", (face_id, similarity_threshold, limit,))
    return c.fetchall()
