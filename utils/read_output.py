# import pickle

# boxes = pickle.load(open('/home/junjie/Code/faster-rcnn-gcn/output/res101/voc_2007_test/rec_bgul/detections.pkl', 'rb'))
# re_class_boxes = pickle.load(open('/home/junjie/Code/faster-rcnn-gcn/output/res101/voc_2007_test/origin/detections.pkl', 'rb'))


# pass


import json
coco = json.load(open('/home/junjie/Code/Datasets/COCO/annotations/instances_trainval2014.json', 'rb'))

class_name = [ 'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor']

class_name = [ 'airplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'dining table', 'dog', 'horse',
                         'motorcycle', 'person', 'potted plant',
                         'sheep', 'couch', 'train', 'tv']


i = 0
index = []
for cat in coco['categories']:
    if cat['name'] in class_name:
        print(cat['name'])
        i += 1
        index.append(cat['id'])
    
print(i)

print(index)
pass