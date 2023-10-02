# EGO-YOLO
We retraine the YOLO-series detection framework on the ego-object dataset in order to obtain a more complete egocentric perspective visual tool chain. The backbone for the detecor is from YOLOv5. The ego-object dataset is from the : https://ai.meta.com/datasets/egoobjects-downloads/. In this work, we do not set the object classification branch in YOLO, only the foreground (object) and background were classified. 
1. We freeze the Classify Decoder and set the classification-head into a binary class structure – front ground and  back ground. 
2. We involve the COCO pretrained backbone and finetune on the Ego-Object Datasets
3. Reset all the data into a COCO format from detron2 format.
Here is the mAP-50 results without pretrained YOLOv5 and pretrained YOLOv5:
<img width="629" alt="image" src="https://github.com/Songluchuan/EGO-YOLO/assets/42260891/2e1d292c-0d23-4591-a886-ccd7ba5579b1">
\textbf{Moreover, we show the real-world (real headset videos) comparison results for reference}
