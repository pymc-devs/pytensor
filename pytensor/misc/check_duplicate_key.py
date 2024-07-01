import pickle
import sys
from pathlib import Path

from pytensor.configdefaults import config


DISPLAY_DUPLICATE_KEYS = False
DISPLAY_MOST_FREQUENT_DUPLICATE_CCODE = False

dirs = []
if len(sys.argv) > 1:
    for compiledir in sys.argv[1:]:
        dirs.extend([x.resolve() for x in Path(compiledir).iterdir()])
else:
    dirs = [x.resolve() for x in config.compiledir.iterdir()]
keys: dict = {}  # key -> nb seen
mods: dict = {}
for dir in dirs:
    if not dir.is_dir():
        continue
    try:
        key = (dir / "key.pkl").read_bytes()
        keys.setdefault(key, 0)
        keys[key] += 1
    except FileNotFoundError:
        # print dir, "don't have a key.pkl file"
        pass
    try:
        path = dir / "mod.cpp"
        if not path.exists():
            path = dir / "mod.cu"
        mod = path.read_text(encoding="utf-8")
        mods.setdefault(mod, ())
        mods[mod] += (key,)
    except FileNotFoundError:
        print(dir, "don't have a mod.{cpp,cu} file")

if DISPLAY_DUPLICATE_KEYS:
    for k, v in keys.items():
        if v > 1:
            print("Duplicate key (%i copies): %s" % (v, pickle.loads(k)))

nbs_keys: dict = {}  # nb seen -> now many key
for val in keys.values():
    nbs_keys.setdefault(val, 0)
    nbs_keys[val] += 1

nbs_mod: dict = {}  # nb seen -> how many key
nbs_mod_to_key = {}  # nb seen -> keys
more_than_one = 0
for mod, kk in mods.items():
    val = len(kk)
    nbs_mod.setdefault(val, 0)
    nbs_mod[val] += 1
    if val > 1:
        more_than_one += 1
    nbs_mod_to_key[val] = kk

if DISPLAY_MOST_FREQUENT_DUPLICATE_CCODE:
    m = max(nbs_mod.keys())
    print("The keys associated to the mod.{cpp,cu} with the most number of copy:")
    for kk in nbs_mod_to_key[m]:
        kk = pickle.loads(kk)
        print(kk)

print("key.pkl histograph")
l = list(nbs_keys.items())
l.sort()
print(l)

print("mod.{cpp,cu} histogram")
l = list(nbs_mod.items())
l.sort()
print(l)

total = sum(len(k) for k in list(mods.values()))
uniq = len(mods)
useless = total - uniq
print("mod.{cpp,cu} total:", total)
print("mod.{cpp,cu} uniq:", uniq)
print("mod.{cpp,cu} with more than 1 copy:", more_than_one)
print("mod.{cpp,cu} useless:", useless, float(useless) / total * 100, "%")

print("nb directory", len(dirs))
