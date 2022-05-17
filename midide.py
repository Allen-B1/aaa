import urllib.request
import http.client
import re

resp: http.client.HTTPResponse = urllib.request.urlopen("http://www.piano-midi.de/")
body = resp.read().decode('utf8')

print(body)
links = re.findall(r"<a href=\"(\w+).htm\" title=\"[\w\s]+Audio Files\">", body)
