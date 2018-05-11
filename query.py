import sqlite3
import json
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument("db")
parser.add_argument("--limit", type=int, default=5)
parser.add_argument("--similarity_threshold", type=float, default=0.35)
subparsers = parser.add_subparsers(dest='command')
common_parser = subparsers.add_parser('get_common_faces')
common_parser.add_argument("--with_gps", action='store_true')
similar_parser = subparsers.add_parser('get_similar_faces')
similar_parser.add_argument("face_id", type=int)
selfies_parser = subparsers.add_parser('get_selfies')
criminals_parser = subparsers.add_parser('get_criminals')
criminals_parser.add_argument("face_id", type=int)
contact_parser = subparsers.add_parser('get_contacts')
args = parser.parse_args()

conn = sqlite3.connect(args.db)
#conn.row_factory = sqlite3.Row

if args.command == 'get_common_faces':
    print(json.dumps(get_common_faces(args.with_gps, args.limit, args.similarity_threshold)))
elif args.command == 'get_similar_faces':
    print(json.dumps(get_similar_faces(args.face_id, args.limit, args.similarity_threshold)))
elif args.command == 'get_selfies':
    print(json.dumps(get_selfies(args.limit, args.similarity_threshold)))
elif args.command == 'get_criminals':
    print(json.dumps(get_criminals(args.face_id, args.limit, args.similarity_threshold)))
