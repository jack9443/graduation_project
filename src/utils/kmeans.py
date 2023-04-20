import tqdm
import numpy as np
import json

from sklearn.cluster import KMeans
from zipfile import ZipFile

PATH = 'data/train/label.zip'

if __name__ == '__main__':
    wh = []
    with ZipFile(PATH) as zf:
        infos = zf.infolist()
        for f in tqdm.tqdm(infos):
            temp = zf.open(f)
            a = json.loads(temp.read())
            wh.append((a['top_coord']['width'], a['top_coord']['height']))
            wh.append((a['bottom_coord']['width'], a['bottom_coord']['height']))

    wh = np.array(wh)
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=500, n_init=10)
    y_pred = kmeans.fit_predict(wh)

    centroids = kmeans.cluster_centers_
    for i, c in enumerate(centroids):
        print(i, c)