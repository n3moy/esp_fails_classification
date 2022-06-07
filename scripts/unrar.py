import os
import patoolib

PATH = "C:\\Users\\vladv\\predictiveAnalytics\\data"

path = []
for dirname, _, filenames in os.walk(PATH):
    for filename in filenames:
        if filename[-4:] == ".rar":
            path.append(os.path.join(dirname, filename))

for p in path:
    patoolib.extract_archive(p, outdir=PATH)