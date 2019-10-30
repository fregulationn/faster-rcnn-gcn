import pickle

boxes = pickle.load(open('/home/junjie/Code/faster-rcnn-gcn/output/res101/voc_2007_test/rec_bgul/detections.pkl', 'rb'))
re_class_boxes = pickle.load(open('/home/junjie/Code/faster-rcnn-gcn/output/res101/voc_2007_test/origin/detections.pkl', 'rb'))


pass