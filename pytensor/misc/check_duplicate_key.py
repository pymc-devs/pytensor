import os
import pickle
import sys
from collections import Counter

from pytensor.configdefaults import config


DISPLAY_DUPLICATE_KEYS = False
DISPLAY_MOST_FREQUENT_DUPLICATE_CCODE = False

dirs = []
if len(sys.argv) > 1:
    for compiledir in sys.argv[1:]:
        dirs.extend([os.path.join(compiledir, d) for d in os.listdir(compiledir)])
else:
    dirs = os.listdir(config.compiledir)
    dirs = [os.path.join(config.compiledir, d) for d in dirs]
keys: Counter[bytes] = Counter()  # key -> nb seen
mods: dict = {}
for dir in dirs:
    key = None
    try:
        with open(os.path.join(dir, "key.pkl"), "rb") as f:
            key = f.read()
        keys[key] += 1
        del f
    except OSError:
        # print dir, "don't have a key.pkl file"
        pass
    try:
        path = os.path.join(dir, "mod.cpp")
        with open(path) as fmod:
            mod = fmod.read()
        mods.setdefault(mod, ())
        mods[mod] += (key,)
        del mod
        del fmod
        del path
    except OSError:
        print(dir, "don't have a mod.cpp file")

if DISPLAY_DUPLICATE_KEYS:
    for k, v in keys.items():
        if v > 1:
            print("Duplicate key (%i copies): %s" % (v, pickle.loads(k)))

# nb seen -> how many keys
nbs_keys = Counter(val for val in keys.values())

# nb seen -> how many keys
nbs_mod = Counter(len(kk) for kk in mods.values())
# nb seen -> keys
nbs_mod_to_key = {len(kk): kk for kk in mods.values()}
more_than_one = sum(len(kk) > 1 for kk in mods.values())

if DISPLAY_MOST_FREQUENT_DUPLICATE_CCODE:
    m = max(nbs_mod)
    print("The keys associated to the mod.cpp with the most number of copy:")
    for kk in nbs_mod_to_key[m]:
        kk = pickle.loads(kk)
        print(kk)

print("key.pkl histograph")
print(sorted(nbs_keys.items()))

print("mod.cpp histogram")
print(sorted(nbs_mod.items()))

total = sum(len(k) for k in mods.values())
uniq = len(mods)
useless = total - uniq
print("mod.cpp total:", total)
print("mod.cpp uniq:", uniq)
print("mod.cpp with more than 1 copy:", more_than_one)
print("mod.cpp useless:", useless, float(useless) / total * 100, "%")

print("nb directory", len(dirs))
