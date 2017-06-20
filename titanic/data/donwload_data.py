import os
import urllib.request

url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
file_path = "titanic3.xls"
if not os.path.isfile(file_path):
    result = urllib.request.urlretrieve(url, file_path)
    print('downloaded:', result)
