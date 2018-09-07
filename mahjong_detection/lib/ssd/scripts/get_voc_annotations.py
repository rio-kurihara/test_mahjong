import pickle
from ssd.utils import VOCAnnotationReader


if __name__ == "__main__":
    file_dir = "../../../share/image-net/VOCdevkit/VOC2007/Annotations"
    labels = ["aeroplane",
              "bus", "bicycle", "bird", "boat", "bottle",
              "car", "cat", "chair", "cow",
              "diningtable", "dog",
              "horse",
              "motorbike",
              "pottedplant", "person",
              "sheep", "sofa",
              "train", "tvmonitor"]

    data = VOCAnnotationReader(file_dir, labels).data
    pickle.dump(data, open("voc2007_annotations.pkl", "wb"))
