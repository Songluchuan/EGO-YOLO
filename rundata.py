import os
import cv2 
import json
import torch
import torchvision
from tqdm import trange

import numpy as np
from pycocotools.coco import COCO
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        # image_id = target["image_id"]
        # image_id = torch.tensor([image_id])
        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # import pdb; pdb.set_trace()
        # print(anno)
        classes = [obj["_category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        # target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder='/home/ptg/egoobjects/egoobject_images', 
					   ann_file='/home/ptg/egoobjects/EgoObjectsV1_unified_eval.json'):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.trans = transforms.Compose([transforms.ToTensor()]) 
        self.prepare = ConvertCocoPolysToMask(False)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        target = {'image_id': img, 'annotations': target}
        _, target = self.prepare(img, target)
        img = self.trans(img)
        # import pdb; pdb.set_trace()
        target = {'annotations': target}
        return img, target

egoCOCO = CocoDetection()
dataloader = DataLoader(egoCOCO)
index = 0
for img, anno in dataloader:
    # bbox = anno['annotations'][0]['bbox']
    
    bboxes = []
    ids = []
    for xx in range(len(anno['annotations']['boxes'][0])):
        label = anno['annotations']['boxes'][0][xx]
        # import pdb;pdb.set_trace()
        bboxes.append([label[0].item(),label[1].item(),label[2].item(),label[3].item()])
        ids.append(anno['annotations']['labels'][0][xx].item())
    np_img = np.array(img[0])*255
    np_img = np.transpose(np_img, (1, 2, 0))

    label_fl = '/home/ptg/object_detector/datasets/egoobject/labels/val2023/%09d.txt'%index

    W = np_img.shape[1]
    H = np_img.shape[0]
    f= open(label_fl, 'w')
    for uu in range(len(bboxes)):
        box = bboxes[uu]
        x1 = (box[0] + box[2])/(2*W)#int(box[0] * W - 0.5 * W*box[2])
        x2 = (box[2] - box[0])/W#int(box[0] * W + 0.5 * W*box[2])
        y1 = (box[1] + box[3])/(2*H)#int(box[1] * H - 0.5 * H*box[3])
        y2 = (box[3] - box[1])/H#int(box[1] * H + 0.5 * H*box[3])
        # x1 = int(box[0] * W - 0.5 * W*box[2])
        # x2 = int(box[0] * W + 0.5 * W*box[2])
        # y1 = int(box[1] * H - 0.5 * H*box[3])
        # y2 = int(box[1] * H + 0.5 * H*box[3])

        # x1 = int(box[0])
        # y1 = int(box[1])
        # x2 = int(box[2])
        # y2 = int(box[3])
        # print(x1, y1, x2, y2)

        strs = '%d %s %s %s %s'%(ids[uu], x1, y1, x2, y2)
        f.write(strs + '\n')

        # 
        # np_img = cv2.rectangle(np_img,(x1,y1),(x2,y2),color=(0,0,255),thickness=2)
    cv2.imwrite('/home/ptg/object_detector/datasets/egoobject/images/val2023/%09d.jpg'%index, np_img[:,:,::-1])
    # cv2.imwrite('/home/ptg/object_detector/YOLOv6/2.jpg', np_img)
    print(index)
    f.close()
    index += 1
    # import pdb;pdb.set_trace()
        

  

	



# for i in trange(len(annotations)):
# 	import pdb; pdb.set_trace()
	
# 	img_name = str(annotations[i]['instance_id'])
# 	img_bbox = annotations[i]['bbox']
# 	img_catg = annotations[i]['category_id']
	
# 	strs = '%s %s %s %s %s'%(img_catg, img_bbox[0], img_bbox[1], img_bbox[2], img_bbox[3])
# 	label_fl = target_label + '/' + img_name+'.txt'
	
# 	with open(label_fl, 'a') as f:
#     		f.write(strs + '\n')
	
# 	name = img_root + '/' + str(annotations[i]['instance_id']) + '.jpg'
# 	target = target_image + '/' + str(annotations[i]['instance_id']) + '.jpg'
# 	cmd = 'cp %s %s'%(name, target)
# 	os.system(cmd)
	
	

#
