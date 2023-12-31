# EGO-YOLO
We retrain the YOLO-series detection framework on the ego-object dataset in order to obtain a more complete egocentric perspective visual tool chain. The backbone for the detecor is from YOLOv5. The ego-object dataset is from the : https://ai.meta.com/datasets/egoobjects-downloads/. In this work, we do not set the object classification branch in YOLO, only the foreground (object) and background were classified. 
1. We freeze the Classify Decoder and set the classification-head into a binary class structure – front ground and  back ground. 
2. We involve the COCO pretrained backbone and finetune on the Ego-Object Datasets
3. Reset all the data into a COCO format from detron2 format.

## How to use it


1. Download the pretrained YOLO: 
The pretrained model is putted in: https://drive.google.com/drive/folders/1j6z27hA8vNA_oCB8aZcYrNG2JDFEJrlu?usp=drive_link , please download the pretrained model (last.pt or best.pt).


2. Install the package:
```pip install -r requirements.txt```


3. Run with:
```python detect.py --weights best.pt --source $Your Image$```



## Some Experimental results
Here is the mAP-50 results without pretrained YOLOv5 and pretrained YOLOv5:
<img width="629" alt="image" src="https://github.com/Songluchuan/EGO-YOLO/assets/42260891/2e1d292c-0d23-4591-a886-ccd7ba5579b1">


## We show the val-set comparison results in below:


The pretrained results:
<img width="892" alt="image" src="https://github.com/Songluchuan/EGO-YOLO/assets/42260891/896b6aba-2609-4be2-97ce-1441aa04efe8">
<img width="889" alt="image" src="https://github.com/Songluchuan/EGO-YOLO/assets/42260891/f9c4f0c6-dd33-4965-aaf1-a2cb7bca9da4">


The origin YOLO results:
<img width="892" alt="image" src="https://github.com/Songluchuan/EGO-YOLO/assets/42260891/ee1bc3c5-00ec-45c9-817a-a01b298c4b1f">
<img width="889" alt="image" src="https://github.com/Songluchuan/EGO-YOLO/assets/42260891/e031d889-dfcb-4040-890a-cb3b00e691c4">


## Then, we show the real-world (real headset videos) comparison results in below:



The pretrained results:


<img width="844" alt="image" src="https://github.com/Songluchuan/EGO-YOLO/assets/42260891/f9c613dd-19cb-424e-a63c-5103ba2f0feb">


The origin YOLO results:
<img width="849" alt="image" src="https://github.com/Songluchuan/EGO-YOLO/assets/42260891/2ed4d9ba-2405-4afe-8e55-64e3cfbb180b">
