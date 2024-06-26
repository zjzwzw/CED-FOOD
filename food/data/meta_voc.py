import os
import numpy as np
import xml.etree.ElementTree as ET
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_voc_instances
from typing import List, Tuple, Union

__all__ = ["register_meta_voc"]


def load_filtered_voc_instances(
     dirname: str, split: str, classnames: str, basecls
):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    name = split
    fileids = {}
    split_dir = os.path.join("datasets", "vocsplit")
    shot = name.split("_")[-2].split("shot")[0]
    seed = int(name.split("_seed")[-1])
    split_dir = os.path.join(split_dir, "seed{}".format(seed))
    for cls in classnames:
        with PathManager.open(
                os.path.join(
                    split_dir, "box_{}shot_{}_train.txt".format(shot, cls)
                )
        ) as f:
            fileids_ = np.loadtxt(f, dtype=np.str).tolist()
            if isinstance(fileids_, str):
                fileids_ = [fileids_]
            fileids_ = [
                fid.split("/")[-1].split(".jpg")[0] for fid in fileids_
            ]
            fileids[cls] = fileids_


    dicts = []
    for cls, fileids_ in fileids.items():
        dicts_ = []
        for fileid in fileids_:
            year = "2012" if "_" in fileid else "2007"
            dirname = os.path.join("datasets", "VOC{}".format(year))
            anno_file = os.path.join(
                dirname, "Annotations", fileid + ".xml"
            )
            jpeg_file = os.path.join(
                dirname, "JPEGImages", fileid + ".jpg"
            )

            tree = ET.parse(anno_file)

            for obj in tree.findall("object"):
                r = {
                    "file_name": jpeg_file,
                    "image_id": fileid,
                    "height": int(tree.findall("./size/height")[0].text),
                    "width": int(tree.findall("./size/width")[0].text),
                }
                cls_ = obj.find("name").text
                if cls != cls_:
                    continue
                bbox = obj.find("bndbox")
                bbox = [
                    float(bbox.find(x).text)
                    for x in ["xmin", "ymin", "xmax", "ymax"]
                ]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                instances = [
                    {
                        "category_id": classnames.index(cls),
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                    }
                ]
                r["annotations"] = instances
                dicts_.append(r)
        if len(dicts_) > int(shot):
            dicts_ = np.random.choice(dicts_, int(shot), replace=False)
        dicts.extend(dicts_)

    return dicts

def load_voc_instances_wocls(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]], basecls):
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if not (cls in class_names):
                continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts

def register_meta_voc(
    name, metadata, dirname, split, year, keepclasses, sid
):
    if keepclasses.startswith("base_novel"):
        thing_classes = metadata["known_classes"][sid]
    elif keepclasses.startswith("base"):
        thing_classes = metadata["base_classes"][sid]
    elif keepclasses.startswith("novel"):
        thing_classes = metadata["novel_classes"][sid]
    elif keepclasses.startswith("all_known_unknown"):
        thing_classes = metadata["thing_classes_21"][sid]

    if "shot" in name:
        func = load_filtered_voc_instances
    else:
        func = load_voc_instances_wocls


    DatasetCatalog.register(
        name,
        lambda: func(
            dirname, split, thing_classes,metadata["base_classes"][sid]
        ),
    )

    MetadataCatalog.get(name).set(
        thing_classes=thing_classes,
        dirname=dirname,
        year=year,
        split=split,
        base_classes=metadata["base_classes"][sid],
        novel_classes=metadata["novel_classes"][sid],
    )
