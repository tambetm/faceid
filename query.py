import facedb as db
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("db")
parser.add_argument("--debug", action='store_true')
parser.add_argument("--limit", type=int, default=5)
parser.add_argument("--similarity_threshold", type=float, default=0.35)
subparsers = parser.add_subparsers(dest='command')
clusters_parser = subparsers.add_parser('get_clusters')
clusters_parser.add_argument("--with_gps", action='store_true')
faces_parser = subparsers.add_parser('get_cluster_faces')
faces_parser.add_argument("cluster_num", type=int)
faces_parser.add_argument("--with_gps", action='store_true')
similar_parser = subparsers.add_parser('get_similar_faces')
similar_parser.add_argument("face_id", type=int)
selfies_parser = subparsers.add_parser('get_selfies')
criminals_parser = subparsers.add_parser('get_criminals')
criminals_parser.add_argument("face_id", type=int)
#contact_parser = subparsers.add_parser('get_contacts')
args = parser.parse_args()

db.connect(args.db, args.debug)

if args.command == 'get_clusters':
    print(json.dumps(db.get_clusters(**vars(args))))
elif args.command == 'get_cluster_faces':
    print(json.dumps(db.get_cluster_faces(**vars(args))))
elif args.command == 'get_similar_faces':
    print(json.dumps(db.get_similar_faces(**vars(args))))
elif args.command == 'get_selfies':
    print(json.dumps(db.get_selfies(**vars(args))))
elif args.command == 'get_criminals':
    print(json.dumps(db.get_criminals(**vars(args))))
