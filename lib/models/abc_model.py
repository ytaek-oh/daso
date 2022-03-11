from .classifier import Classifier
from .semi_model import SemiModel


class ABCModel(SemiModel):

    def __init__(self, cfg):
        super(ABCModel, self).__init__(cfg)
        self.abc_classifier = Classifier(self.out_features, self.num_classes)

    def forward(self, x, is_train=True, return_features=False):
        x = self.encoder(x)
        if return_features:
            return x

        if not is_train:
            return self.abc_classifier(x)  # for evaluation, use aux classifier

        # by default, linear classification
        return self.classifier(x)

    def classify(self, x):
        return self.classifier(x)

    def abc_classify(self, x):
        return self.abc_classifier(x)

    def rotate_classify(self, x):
        return self.projection(x)
