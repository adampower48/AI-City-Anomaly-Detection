import PIL
import sklearn.preprocessing as prep
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn

from reid import misc, modelling


class ReidExtractor:
    """Base feature extractor class.
    args:
        features: List of features.
    """


    def __init__(self, model_name, model_path, image_size=320):
        model = modelling.build_model(model_name, 2000)
        model.cuda()
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path), strict=False)
        model.eval()
        self.extractor = model
        self.image_size = image_size
        self.transforms = misc.preprocess(misc.normalize_torch, self.image_size)


    def extract(self, img_numpy, region):
        img = PIL.Image.fromarray(img_numpy)
        img = img.crop(region)

        model_input = self.transforms(img).unsqueeze(0)

        with torch.no_grad():
            features_after, features_before = self.extractor(model_input.cuda())

        features = prep.normalize(features_before.cpu().data.numpy(), norm="l2").astype("float32")

        return features


    def similarity(self, img1, img2, region1, region2):
        return cosine_similarity(self.extract(img1, region1),
                                 self.extract(img2, region2))
