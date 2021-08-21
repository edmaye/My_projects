import argparse
import base64
import json
import os,glob
import os.path as osp

import imgviz
import PIL.Image

from labelme.logger import logger
from labelme import utils


def main():
    logger.warning(
        "This script is aimed to demonstrate how to convert the "
        "JSON file to a single image dataset."
    )
    logger.warning(
        "It won't handle multiple JSON files to generate a "
        "real-use dataset."
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    parser.add_argument("--out", default='./output')
    args = parser.parse_args()

    json_file = args.json_file


    out_dir = args.out
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    json_list = glob.glob(os.path.join(json_file,'*.json'))
    for json_path in json_list:
        try:
            id = os.path.basename(json_path).split('.')[0]
            
            data = json.load(open(json_path))
            imageData = data.get("imageData")

            if not imageData:
                imagePath = os.path.join(os.path.dirname(json_path), data["imagePath"])
                with open(imagePath, "rb") as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode("utf-8")
            img = utils.img_b64_to_arr(imageData)

            label_name_to_value = {"_background_": 0}
            for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                label_name = shape["label"]
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            lbl, _ = utils.shapes_to_label(
                img.shape, data["shapes"], label_name_to_value
            )

            label_names = [None] * (max(label_name_to_value.values()) + 1)
            for name, value in label_name_to_value.items():
                label_names[value] = name

            lbl_viz = imgviz.label2rgb(
                label=lbl, img=imgviz.asgray(img), label_names=label_names, loc="rb"
            )


            utils.lblsave(osp.join(out_dir, id+".png"), lbl)

            # 保存
            #PIL.Image.fromarray(img).save(osp.join(out_dir, id+"_img.png"))
            #utils.lblsave(osp.join(out_dir, id+"_label.png"), lbl)
            # PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, id+"_label_viz.png"))

            # with open(osp.join(out_dir, id+"_label_names.txt"), "w") as f:
            #     for lbl_name in label_names:
            #         f.write(lbl_name + "\n")


            logger.info(json_path+" saved to: {}".format(out_dir))
        except:
            print('ERROR:',json_path)


if __name__ == "__main__":
    main()
