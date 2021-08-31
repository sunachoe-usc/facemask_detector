from typing import Union

import cv2
import os
import numpy as np
import random
from itertools import chain
from matplotlib import pyplot as plt
from xml.etree import ElementTree

from numpy import ndarray

count=0
count1=0
lenlist = []
newvar = []
all_newvar = []
labels1 = []
labels2 = []
labels3 = []
labels4 = []
cropped_faces_list = []
all_cropped_faces_list = []
new_vectors = []
all_new_vectors = []
duplicate_randomlist = []
#randomlist = []


# for r in range(0,10):
#     ran = random.sample(0,11)
#     randomlist.append(ran)

for x in range(0,327):#160
    random.seed(x*10)
    duplicate_randomlist.append(random.randint(0,3275)) #3208

duplicate_randomlist.sort()
randomlist = []
for i in duplicate_randomlist:
    if i not in randomlist:
        randomlist.append(i)

#array=numpy.zeros(shape=(7500,853))
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
        all_new_vectors.append(new)
        #if os.stat(new).st_size > 5000:
        cv2.imwrite(os.path.join('../Cropped_faces', 'cropped' + str(count) + '.png'), new)
        #cropped_faces_list.append('cropped'+str(count)+'.png')
        cropped_faces_list.append(count)
        all_cropped_faces_list.append(count)
        newvar.append(count1)
        all_newvar.append(count1)

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

        all_newf = all_new_vectors[-1].flatten()
        all_newvar[count] = all_newf
        ab=np.vstack(all_newvar)

        count += 1
print(aa.shape)
print(ab.shape)

for h in range(0,len(labels1)):
    labels3.append('x')
    labels4.append('x')


for k in range(0,len(randomlist)):
    labels3[randomlist[k]] = cropped_faces_list[randomlist[k]]
    labels4[randomlist[k]] = labels1[randomlist[k]]

#print(labels3)
#print(labels4)


aa = np.float32(aa)
clusters = 23
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(aa,clusters,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)


distlists = []
centroids = []
centroid_distances = []
labels_centroids = []
compare = []
labeled_data_count = []
labeled_data_in_each_centroid_imgnum = []
labeled_data_in_each_centroid_imgnum_1 = []
random_distlists = []
labeled_data_in_each_centroid = []
label_count = []

for v in range(0,clusters):
    distlists.append([])
    centroids.append([])
    centroid_distances.append([])
    labels_centroids.append([])
    labeled_data_count.append([])
    labeled_data_in_each_centroid_imgnum.append([])
    labeled_data_in_each_centroid_imgnum_1.append([])
    random_distlists.append([])
    labeled_data_in_each_centroid.append([])
    label_count.append([])

for p in range(0,clusters):
    labeled_data_count[p].append(0)


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


# for t in range(0,clusters):
#     a = labels_centroids[t].count('with_mask')
#     b = labels_centroids[t].count('without_mask')
#     c = labels_centroids[t].count('mask_weared_incorrect')
#     print('In centroid',t,':',a, b, c)
    #print('fraction:', a/(a+b+c), b/(a+b+c), c/(a+b+c))
    #print('total:',len(centroids[t]))

#print(centroids)



for o in range(0,clusters):
    for r in range(0,len(centroids[o])):
        if centroids[o][r] in labels3:
            labeled_data_count[o][0] = labeled_data_count[o][0]+1
            labeled_data_in_each_centroid_imgnum[o].append(centroids[o][r])
    #print(labeled_data_count[o])
    for e in range(0,len(labeled_data_in_each_centroid_imgnum[o])):
        labeled_data_in_each_centroid[o].append(labels2[labeled_data_in_each_centroid_imgnum[o][e]])
    w_mask = labeled_data_in_each_centroid[o].count('with_mask')
    wo_mask = labeled_data_in_each_centroid[o].count('without_mask')
    incorrect_mask = labeled_data_in_each_centroid[o].count('mask_weared_incorrect')
    label_count[o].append(w_mask)
    label_count[o].append(wo_mask)
    label_count[o].append(incorrect_mask)


#print('labeled_data_in_each_centroid_imgnum',labeled_data_in_each_centroid_imgnum)

print('label_count',label_count)


for u in range(0,clusters):
    for x in range(0,3):
        labeled_data_in_each_centroid_imgnum_1[u].append([])
        random_distlists[u].append([])
    for y in range(0,len(labeled_data_in_each_centroid_imgnum[u])):
        if labels2[labeled_data_in_each_centroid_imgnum[u][y]]== 'with_mask':
            labeled_data_in_each_centroid_imgnum_1[u][0].append(labeled_data_in_each_centroid_imgnum[u][y])
        if labels2[labeled_data_in_each_centroid_imgnum[u][y]] == 'without_mask':
            labeled_data_in_each_centroid_imgnum_1[u][1].append(labeled_data_in_each_centroid_imgnum[u][y])
        if labels2[labeled_data_in_each_centroid_imgnum[u][y]] == 'mask_weared_incorrect':
            labeled_data_in_each_centroid_imgnum_1[u][2].append(labeled_data_in_each_centroid_imgnum[u][y])



#print('labeled_data_in_each_centroid_imgnum_1',labeled_data_in_each_centroid_imgnum_1)
# for l in range(0,len(labeled_data_in_each_centroid_imgnum_1)):
#     for e in range(0,len(labeled_data_in_each_centroid_imgnum_1[l])):
#         for n in range(0,len(labeled_data_in_each_centroid_imgnum_1[l][e])):
#             print('n',len(labeled_data_in_each_centroid_imgnum_1[l][e]))




# flatten_labeled_data_in_each_centroid = list(chain.from_iterable(labeled_data_in_each_centroid))
# #print(flatten_labeled_data_in_each_centroid)
# w_mask = flatten_labeled_data_in_each_centroid.count('with_mask')
# wo_mask = flatten_labeled_data_in_each_centroid.count('without_mask')
# incorrect_mask = flatten_labeled_data_in_each_centroid.count('mask_weared_incorrect')
# print('w',w_mask,wo_mask,incorrect_mask)
# #print(w_mask+wo_mask+incorrect_mask)
# x_1 = w_mask/(w_mask+wo_mask+incorrect_mask)
# y_1 = wo_mask/(w_mask+wo_mask+incorrect_mask)
# z_1 = incorrect_mask/(w_mask+wo_mask+incorrect_mask)
# print(x_1, y_1, z_1)
#
# list_f_x = []
# list_f_y = []
# list_f_z = []
# for f in range(0,len(label_count)):
#     f_x = label_count[f][0]/(sum(label_count[f]))
#     f_y = label_count[f][1]/(sum(label_count[f]))
#     f_z = label_count[f][2] / (sum(label_count[f]))
#     list_f_x.append(f_x)
#     list_f_y.append(f_y)
#     list_f_z.append(f_z)
#
# print('list_f_x',list_f_x)
# print('list_f_y',list_f_y)
# print('list_f_z',list_f_z)
#
# for w in range(len(list_f_x)):
#     weight_x = list_f_x[w]/x_1
#     print('weight_x in cluster',w,':',weight_x)
#     # weight_y = list_f_y[w]/y_1
#     # print('weight_y in cluster', w, ':', weight_y)
#     weight_z = list_f_z[w]/z_1
#     print('weight_z in cluster', w, ':', weight_z)


#f_y =
#f_z =

# print('labeled_data_in_each_centroid_imgnum',labeled_data_in_each_centroid_imgnum)
# print('labeled_data_in_each_centroid',labeled_data_in_each_centroid)

# a_count=0
# m_count=0
# b_count=0
# for d in range(0,len(randomlist)):
#     compare.append([])
for m in range(0,clusters):
    for a in range(0,3):
        for b in range(0,len(labeled_data_in_each_centroid_imgnum_1[m][a])):
            #labeled_data_in_each_centroid_imgnum_1[m][a][b]
            #print(labeled_data_in_each_centroid_imgnum_1[m][a][b])
            dist1 = int(np.linalg.norm(ab[labeled_data_in_each_centroid_imgnum_1[m][a][b]]-center[m]))
            #print('s',int(np.linalg.norm(aa[labeled_data_in_each_centroid_imgnum_1[m][a][b]]-center[m])))

            random_distlists[m][a].append(dist1)

                #random_distlists[m][a][b]=dist1
#print(random_distlists)
                #dist = np.linalg.norm(aa[labeled_data_in_each_centroid_imgnum_1[m][a][b]]-center[m])
                #random_distlists[m][a].append(dist)
#print('d',labeled_data_in_each_centroid_imgnum_1[0])
        # dist = np.linalg.norm(aa[d]-center[m])
        # distlists[m].append(dist)
        # compare[d].append(distlists[m][d])

sum_w_mask = []
sum_wo_mask = []
sum_incorrect_mask = []
for d in random_distlists:
    if len(d[0]) == 0:
        sum_w_mask.append(0)
    else:
        sum_w_mask.append(sum(d[0])/len(d[0]))
    if len(d[1]) == 0:
        sum_wo_mask.append(0)
    else:
        sum_wo_mask.append(sum(d[1])/len(d[1]))
    if len(d[2]) == 0:
        sum_incorrect_mask.append(0)
    else:
        sum_incorrect_mask.append(sum(d[2])/len(d[2]))
    #for i in range(0,3):
        #print(sum(distlists[d][0]))
#print('w_mask average distance in each cluster',sum_w_mask)
#print('wo_mask average distance in each cluster',sum_wo_mask)
#print('incorrect_mask average distance in each cluster',sum_incorrect_mask)

#print('distances',random_distlists)

#w_mask_average_distance = sum(sum_w_mask)/len(sum_w_mask)
#wo_mask_average_distance = sum(sum_wo_mask)/len(sum_wo_mask)
#incorrect_mask_average_distance = sum(sum_incorrect_mask)/len(sum_incorrect_mask)
#print(sum_w_mask, sum(sum_w_mask), 'average distance',w_mask_average_distance)
#print(sum_wo_mask, sum(sum_wo_mask), 'average distance',wo_mask_average_distance)
#print(sum_incorrect_mask, sum(sum_incorrect_mask), 'average distance',incorrect_mask_average_distance)

flatten_labeled_data_in_each_centroid = list(chain.from_iterable(labeled_data_in_each_centroid))
#print(flatten_labeled_data_in_each_centroid)
w_mask = flatten_labeled_data_in_each_centroid.count('with_mask')
wo_mask = flatten_labeled_data_in_each_centroid.count('without_mask')
incorrect_mask = flatten_labeled_data_in_each_centroid.count('mask_weared_incorrect')
print('w',w_mask,wo_mask,incorrect_mask)
#print(w_mask+wo_mask+incorrect_mask)
x_1 = w_mask/(w_mask+wo_mask+incorrect_mask)
y_1 = wo_mask/(w_mask+wo_mask+incorrect_mask)
z_1 = incorrect_mask/(w_mask+wo_mask+incorrect_mask)
print(x_1, y_1, z_1)
print('total labels',w_mask+wo_mask+incorrect_mask)

for t in range(0,clusters):
    a = labels_centroids[t].count('with_mask')
    b = labels_centroids[t].count('without_mask')
    c = labels_centroids[t].count('mask_weared_incorrect')
    print('In centroid',t,':',a, b, c)

list_f_x = []
list_f_y = []
list_f_z = []
for f in range(0,len(label_count)):
    try:
        f_x = label_count[f][0]/(sum(label_count[f]))
    except ZeroDivisionError:
        f_x = 0
    try:
        f_y = label_count[f][1]/(sum(label_count[f]))
    except ZeroDivisionError:
        f_y = 0
    try:
        f_z = label_count[f][2]/(sum(label_count[f]))
    except ZeroDivisionError:
        f_z = 0

    list_f_x.append(f_x)
    list_f_y.append(f_y)
    list_f_z.append(f_z)

# print('list_f_x',list_f_x)
# print('list_f_y',list_f_y)
# print('list_f_z',list_f_z)

for w in range(len(list_f_x)):
    try:
        weight_x = list_f_x[w]/x_1
    except ZeroDivisionError:
        weight_x = 0
    try:
        weight_y = list_f_y[w] / y_1
    except ZeroDivisionError:
        weight_y = 0
    try:
        weight_z = list_f_z[w] / z_1
    except ZeroDivisionError:
        weight_z = 0
    if weight_x == 0:
        w_mask_weighted_average_distance = 0
    else:
        w_mask_weighted_average_distance = sum_w_mask[w]/weight_x
    if weight_y == 0:
        wo_mask_weighted_average_distance = 0
    else:
        wo_mask_weighted_average_distance = sum_wo_mask[w]/weight_y
    if weight_z == 0:
        incorrect_mask_weighted_average_distance = 0
    else:
        incorrect_mask_weighted_average_distance = sum_incorrect_mask[w]/weight_z

    #print(sum_w_mask[w], weight_x)
    #print(list_f_x[w], x_1)
    print('weighted average distance for with_mask in cluster',w,':', w_mask_weighted_average_distance)
    #weight_y = list_f_y[w]/y_1
    #wo_mask_weighted_average_distance = wo_mask_average_distance/weight_y
    print('weighted average distance for without_mask in cluster', w, ':', wo_mask_weighted_average_distance)
    #weight_z = list_f_z[w]/z_1
    #incorrect_mask_weighted_average_distance = incorrect_mask_average_distance/weight_z
    print('weighted average distance for incorrect_mask in cluster', w, ':', incorrect_mask_weighted_average_distance)





print(len(centroids))
print(len(centroid_distances))
print(len(cropped_faces_list))
print(len(all_cropped_faces_list))
print(len(labels1))
print(len(labels2))


# for z in range(0,clusters):
#     print('Shortest distance in cluster',z,':',min(centroid_distances[z]))

# for s in range(0,clusters):
#     print('Image number for cluster',s,':',centroids[s][centroid_distances[s].index(min(centroid_distances[s]))])
# for p in range(0,clusters):
#     print('Label for closest image to centroid',p,':',labels2[centroids[p][centroid_distances[p].index(min(centroid_distances[p]))]])




print('count:',count)
print('count1:',count1)


#print(duplicate_randomlist)
#print(randomlist)
print(len(duplicate_randomlist))
print(len(randomlist))