import os
from argparse import ArgumentParser
import cv2
import sys
import glob as go
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'../'))
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import numpy as np
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import scipy.misc

def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default="/opt/data/private/img_dir/natural/NC2016/val/image/",help='Image file')
    parser.add_argument('--config', default="/opt/data/private/sunyu/model/NC2016/B-384-1-1-concatedge_93.03/"
                "SETR_our_mla.py",help='Config file')
    parser.add_argument('--checkpoint', default="/opt/data/private/sunyu/model/NC2016/B-384-1-1-concatedge_93.03/iter_30000.pth",help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    # model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    # result = inference_segmentor(model, args.img)
    # show the results
    # show_result_pyplot(model, args.img, result, get_palette(args.palette))
    demo_dir = "/opt/data/private/sunyu/result/NC2016/B-384-1-1-concatedge_93.03/"  #### val_result  test_data
    img_list = os.listdir(args.img)
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    for num_index, img_index in enumerate(img_list):
       # if(num_index == 2):
       #     break

       # if(num_index % 10 == 0):
       #     print("num_index: ",num_index)
        img_path = os.path.join(args.img, img_index)
        demo_path = os.path.join(demo_dir, img_index[:-3]+"png")
        # build the model from a config file and a checkpoint file
        # test a single image
        result = inference_segmentor(model, img_path)

        result_gray = result[0]
        result_gray[result_gray==1] = 255
        result_gray = result_gray.astype(np.uint8)
        im = Image.fromarray(result_gray)
        im.save(demo_path)


if __name__ == '__main__':
    main()
    true_path = sorted(go.glob("/opt/data/private/img_dir/natural/NC2016/val/gt/*.png"))
    pred_path = sorted(go.glob("/opt/data/private/sunyu/result/NC2016/B-384-1-1-concatedge_93.03/*.png"))
    length = len(true_path)
    image_tp = 0
    image_fp = 0
    image_fn = 0
    pixel_tp = 0
    pixel_fp = 0
    pixel_fn = 0
    for i in range(length):
        y_true = cv2.imread(true_path[i])
        y_pred = cv2.imread(pred_path[i]) / 255
        y_true = y_true[:, :, 1]
        y_pred = y_pred[:, :, 1]
        a = np.sum(y_true)
        b = np.sum(y_pred)
        if a > 0 and b > 0:
            image_tp = image_tp + 1
        if a == 0 and b > 0:
            image_fp = image_fp + 1
        if a > 0 and b == 0:
            image_fn = image_fn + 1
        w = y_true.shape[0]
        h = y_true.shape[1]
        array1 = np.ones((w, h)) * 0
        for ii in range(w):
            for jj in range(h):
                if y_true[ii, jj] == 1 and y_pred[ii, jj] == 1:
                    array1[ii, jj] = 1
        pixel_tp = pixel_tp + np.sum(array1)
        array2 = y_pred - array1
        pixel_fp = pixel_fp + np.sum(array2)
        array3 = y_true - array1
        pixel_fn = pixel_fn + np.sum(array3)

    pixel_precision = pixel_tp / (pixel_tp + pixel_fp)
    pixel_recall = pixel_tp / (pixel_tp + pixel_fn)
    pixel_f1score = 2 * pixel_precision * pixel_recall / (pixel_precision + pixel_recall)
    print('pixel_precision=', pixel_precision)
    print('pixel_recall=', pixel_recall)
    print('pixel_f1score=', pixel_f1score)

    image_precision = image_tp / (image_tp + image_fp)
    image_recall = image_tp / (image_tp + image_fn)
    image_f1score = 2 * image_precision * image_recall / (image_precision + image_recall)
    print('image_precision=', image_precision)
    print('image_recall=', image_recall)
    print('image_f1score=', image_f1score)
