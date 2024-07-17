import pickle
import sys
from collections import Counter
from pathlib import Path

from pytensor.configdefaults import config


DISPLAY_DUPLICATE_KEYS = False
DISPLAY_MOST_FREQUENT_DUPLICATE_CCODE = False

dirs: list = []
if len(sys.argv) > 1:
    for compiledir in sys.argv[1:]:
        dirs.extend(x.resolve() for x in Path(compiledir).iterdir())
else:
    dirs = [x.resolve() for x in config.compiledir.iterdir()]
keys: Counter[bytes] = Counter()  # key -> nb seen
mods: dict = {}
for dir in dirs:
    if not dir.is_dir():
        continue
    try:
        key = (dir / "key.pkl").read_bytes()
        keys[key] += 1
    except FileNotFoundError:
        # print dir, "don't have a key.pkl file"
        pass
    try:
        mod = (dir / "mod.cpp").read_text(encoding="utf-8")
        mods.setdefault(mod, ())
        mods[mod] += (key,)
    except FileNotFoundError:
        print(dir, "don't have a mod.cpp file")

if DISPLAY_DUPLICATE_KEYS:
    for k, v in keys.items():
        if v > 1:
            print(f"Duplicate key ({v} copies): {pickle.loads(k)}")

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
