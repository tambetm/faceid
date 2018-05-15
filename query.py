import facedb as db
import json
import argparse

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

db.connect(args.db)

if args.command == 'get_common_faces':
    print(json.dumps(db.get_common_faces(args.with_gps, args.limit, args.similarity_threshold)))
elif args.command == 'get_similar_faces':
    print(json.dumps(db.get_similar_faces(args.face_id, args.limit, args.similarity_threshold)))
elif args.command == 'db.get_selfies':
    print(json.dumps(db.get_selfies(args.limit, args.similarity_threshold)))
elif args.command == 'get_criminals':
    print(json.dumps(db.get_criminals(args.face_id, args.limit, args.similarity_threshold)))
