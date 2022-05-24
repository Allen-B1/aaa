from typing import List
import urllib.request
import http.client
import re
import zipfile
import os
import io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--skip-download", action='store_true', help="skip downloading .midi files")
args = parser.parse_args()

if not args.skip_download:
    try:
        os.makedirs("datasets/midide")
    except FileExistsError: pass
    with open("datasets/midide/README", "w") as f:
        f.write("Dataset scraped from http://www.piano-midi.de/, licensed under CC-BY-SA\n")

    resp: http.client.HTTPResponse = urllib.request.urlopen("http://www.piano-midi.de/")
    body = resp.read().decode('utf8')
    composers = re.findall(r"<a href=\"(\w+).htm\" title=\"[\w\s]+ Midi and Audio Files\">", body)
    composers_zip: List[str] = []
    for id in composers:
        resp = urllib.request.urlopen("http://www.piano-midi.de/" + id + ".htm")
        body = resp.read().decode('utf8')
        result = re.findall(r"<a href=\"zip/(\w+).zip\">", body)
        if len(result) != 0:
            composers_zip.append(result[0])
        else: # no zip file
            pieces = re.findall(r"href=\"(midis/\w+/(\w+)_format0.mid)\">♫</a></td>", body)
            print("collecting without zip " + id + "...")
            for (path, name) in pieces:
                try:
                    os.mkdir("datasets/midide/" + id)
                except FileExistsError: pass

                resp_midi: http.client.HTTPResponse = urllib.request.urlopen("http://piano-midi.de/" + path)
                with open("datasets/midide/" + id + "/" + name+ ".mid", "wb") as bf:
                    bf.write(resp_midi.read())
            
    for id in composers_zip:
        print("collecting "+ id + ".zip ...")
        try:
            resp_zip: http.client.HTTPResponse = urllib.request.urlopen("http://www.piano-midi.de/zip/" + id + ".zip")
            zip_data = resp_zip.read()
            zip_file = zipfile.ZipFile(io.BytesIO(zip_data))
            os.mkdir("datasets/midide/" + id)
            zip_file.extractall("datasets/midide/" + id)
        except FileExistsError:
            print(id + ".zip already exists")

# convert to MusicXML
import glob
import os.path
for filename in glob.glob("datasets/midide/**/*.mid*"):
    print("converting " + filename + " to MusicXML...")
    composer = os.path.basename(os.path.dirname(filename))
    basename = os.path.splitext(os.path.basename(filename))[0]
    os.system("flatpak run org.musescore.MuseScore -o '%s' '%s'" % ("datasets/midide/" + composer + "/" + basename + ".musicxml", filename))
