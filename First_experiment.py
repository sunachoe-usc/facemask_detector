import cv2
import os
import numpy as np
from xml.etree import ElementTree

count=0
count1=0
lenlist = []
newvar = []
labels1 = []
labels2 = []
cropped_faces_list = []
all_cropped_faces_list = []
new_vectors = []

for i in range(0, 853):
    newString = "maskimages/maksssksksss" + str(i) + ".png"
    img = cv2.imread(newString)

    a_list = []
    b_list = []
    c_list = []
    d_list = []

    newxml = "maskannotations/maksssksksss" + str(i) + ".xml"
    full_file = os.path.abspath(os.path.join(newxml))
    dom = ElementTree.parse(full_file)
    xmin = dom.findall('object/bndbox/xmin')
    ymin = dom.findall('object/bndbox/ymin')
    xmax = dom.findall('object/bndbox/xmax')
    ymax = dom.findall('object/bndbox/ymax')

    object = dom.findall('object')
    lengthobject = len(object)
    lenlist.append(len(object))
    lab = dom.findall('object/name')
    for l in lab:
        labels1.append(l.text)
        labels2.append(l.text)

    for a in xmin:
        a_list.append(a.text)

    for b in ymin:
        b_list.append(b.text)

    for c in xmax:
        c_list.append(c.text)

    for d in ymax:
        d_list.append(d.text)


    for x in range(0, lenlist[i]):
        crop_face = img[int(b_list[x]):int(d_list[x]), int(a_list[x]):int(c_list[x])]
        new = cv2.resize(crop_face, (1, 1), interpolation=cv2.INTER_NEAREST)
        new_vectors.append(new)
        cv2.imwrite(os.path.join('../Cropped_faces', 'cropped' + str(count) + '.png'), new)
        cropped_faces_list.append(count)
        all_cropped_faces_list.append(count)
        newvar.append(count1)

        if os.stat('Cropped_faces/cropped'+str(count)+'.png').st_size < 1:
            os.remove('Cropped_faces/cropped'+str(count)+'.png')
            labels1.pop()
            cropped_faces_list.remove(count)
            newvar.pop()
            new_vectors.pop()

        else:
            newf = new_vectors[-1].flatten()
            newvar[count1] = newf
            aa=np.vstack(newvar)
            count1 += 1
        count += 1

aa = np.float32(aa)
clusters = 22
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(aa,clusters,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

print('aa',aa.shape)
print('len',len(center[0]))
distlists = []
centroids = []
centroid_distances = []
labels_centroids = []
compare = []

for v in range(0,clusters):
    distlists.append([])
    centroids.append([])
    centroid_distances.append([])
    labels_centroids.append([])

for d in range(0,len(cropped_faces_list)):
    compare.append([])
    for m in range(0,clusters):
        dist = np.linalg.norm(aa[d]-center[m])
        distlists[m].append(dist)
        compare[d].append(distlists[m][d])

for g in range(0,len(cropped_faces_list)):
    for n in range(0,clusters):
        if min(compare[g]) == compare[g][n]:
            centroids[n].append(cropped_faces_list[g])
            centroid_distances[n].append(distlists[n][g])

for m in range(0,clusters):
    for u in centroids[m]:
        labels_centroids[m].append(labels2[u])


for t in range(0,clusters):
    print('In centroid',t,':',labels_centroids[t].count('with_mask'), labels_centroids[t].count('without_mask'), labels_centroids[t].count('mask_weared_incorrect'),labels2[centroids[t][centroid_distances[t].index(min(centroid_distances[t]))]])

# print(len(centroids))
# print(len(centroid_distances))
# print(len(cropped_faces_list))
# print(len(all_cropped_faces_list))
# print(len(labels1))
# print(len(labels2))

for z in range(0,clusters):
    print('Shortest distance in cluster',z,':',min(centroid_distances[z]))

for s in range(0,clusters):
    print('Image number for cluster',s,':',centroids[s][centroid_distances[s].index(min(centroid_distances[s]))])

print('count:',count)
print('count1:',count1)

print(aa.shape)
