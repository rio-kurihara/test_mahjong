import json
import keras
from .models import SSD300_vgg16, SSD512_vgg16
from .models import SSD300_resnet50, SSD512_resnet50
from .models import SSD300_xception
from .losses import MultiBoxLoss
from .utils import BoundaryBox


class SingleShotMultiBoxDetector:
    """
    """
    available_type = ["ssd300", "ssd512"]
    available_net = ["vgg16", "resnet50", "xception"]

    ar_presets = dict(
        ssd300=[[2., 1/2.],
                [2., 1/2., 3., 1/3.],
                [2., 1/2., 3., 1/3.],
                [2., 1/2., 3., 1/3.],
                [2., 1/2.],
                [2., 1/2.]],
        ssd512=[[2., 1/2.],
                [2., 1/2., 3., 1/3.],
                [2., 1/2., 3., 1/3.],
                [2., 1/2., 3., 1/3.],
                [2., 1/2., 3., 1/3.],
                [2., 1/2.],
                [2., 1/2.]]
    )
    scale_presets = dict(
        ssd300=[(30., 60.),
                (60., 111.),
                (111., 162.),
                (162., 213.),
                (213., 264.),
                (264., 315.)],
        ssd512=[(20.48, 51.2),
                (51.2, 133.12),
                (133.12, 215.04),
                (215.04, 296.96),
                (296.96, 378.88),
                (378.88, 460.8),
                (460.8, 542.72)]
    )
    default_shapes = dict(
        ssd300=(300, 300, 3)
    )

    def __init__(self, n_classes=1, class_names=["bg"], input_shape=None,
                 aspect_ratios=None, scales=None, variances=None,
                 overlap_threshold=0.5, nms_threshold=0.45,
                 max_output_size=400,
                 model_type="ssd300", base_net="vgg16"):
        """
        """
        self.n_classes = n_classes
        self.class_names = class_names
        if "bg" != class_names[0]:
            print("Warning: Fist label should be bg."
                  " It'll be added automatically.")
            self.class_names = ["bg"] + class_names
            self.n_classes += 1
        if input_shape:
            self.input_shape = input_shape
        else:
            self.input_shape = self.default_shapes[model_type]
        if aspect_ratios:
            self.aspect_ratios = aspect_ratios
        else:
            self.aspect_ratios = self.ar_presets[model_type]
        if scales:
            self.scales = scales
        else:
            self.scales = self.scale_presets[model_type]
        if variances:
            self.variances = variances
        else:
            self.variances = [0.1, 0.1, 0.2, 0.2]
        self.overlap_threshold = overlap_threshold
        self.nms_threshold = nms_threshold
        self.max_output_size = max_output_size
        self.model_type = model_type
        self.base_net = base_net
        self.preprocesser = None
        if base_net == "vgg16":
            from keras.applications.vgg16 import preprocess_input
        elif base_net == "resnet50":
            from keras.applications.resnet50 import preprocess_input
        elif base_net == "xception":
            from keras.applications.xception import preprocess_input
        else:
            raise TypeError("Unknown base net name.")
        self.preprocesser = preprocess_input

        self.model = None
        self.bboxes = None

    def build(self, init_weight="keras_imagenet"):
        """
        """
        # create network
        if self.model_type == "ssd300" and self.base_net == "vgg16":
            self.model, priors = SSD300_vgg16(self.input_shape,
                                              self.n_classes,
                                              self.aspect_ratios,
                                              self.scales)
        elif self.model_type == "ssd300" and self.base_net == "resnet50":
            self.model, priors = SSD300_resnet50(self.input_shape,
                                                 self.n_classes,
                                                 self.aspect_ratios,
                                                 self.scales)
        elif self.model_type == "ssd300" and self.base_net == "xception":
            self.model, priors = SSD300_xception(self.input_shape,
                                                 self.n_classes,
                                                 self.aspect_ratios,
                                                 self.scales)
        elif self.model_type == "ssd512" and self.base_net == "vgg16":
            self.model, priors = SSD512_vgg16(self.input_shape,
                                              self.n_classes,
                                              self.aspect_ratios,
                                              self.scales)
        elif self.model_type == "ssd512" and self.base_net == "resnet50":
            self.model, priors = SSD512_resnet50(self.input_shape,
                                                 self.n_classes,
                                                 self.aspect_ratios,
                                                 self.scales)
        else:
            raise NameError(
                "{},{} is not defined. types are {}, basenets are {}.".format(
                    self.model_type, self.base_net,
                    self.available_type, self.available_net
                )
            )

        if init_weight is None:
            print("Network has not initialized with any pretrained models.")
        elif init_weight == "keras_imagenet":
            print("Initializing network with keras application model"
                  " pretrained imagenet.")
            if self.base_net == "vgg16":
                import keras.applications.vgg16 as keras_vgg16
                weights_path = keras_vgg16.get_file(
                    'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    keras_vgg16.WEIGHTS_PATH_NO_TOP,
                    cache_subdir="models"
                )
            elif self.base_net == "resnet50":
                import keras.applications.resnet50 as keras_resnet50
                weights_path = keras_resnet50.get_file(
                    'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    keras_resnet50.WEIGHTS_PATH_NO_TOP,
                    cache_subdir="models"
                )
            elif self.base_net == "xception":
                import keras.applications.xception as keras_xception
                weights_path = keras_xception.get_file(
                    'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    keras_xception.TF_WEIGHTS_PATH_NO_TOP,
                    cache_subdir="models"
                )
            else:
                raise NameError(
                    "{} is not defined.".format(
                        self.base_net
                    )
                )
            self.model.load_weights(weights_path, by_name=True)
        else:
            print("Initializing network from file {}.".format(init_weight))
            self.model.load_weights(init_weight, by_name=True)

        # make boundary box class
        self.bboxes = BoundaryBox(n_classes=self.n_classes,
                                  default_boxes=priors,
                                  variances=self.variances,
                                  overlap_threshold=self.overlap_threshold,
                                  nms_threshold=self.nms_threshold,
                                  max_output_size=self.max_output_size)

    def train_by_generator(self, gen, epoch=30, neg_pos_ratio=3.0,
                           learning_rate=1e-3, freeze=None, checkpoints=None,
                           optimizer=None):
        """
        """
        # set freeze layers
        if freeze is None:
            freeze = list()

        for L in self.model.layers:
            if L.name in freeze:
                L.trainable = False

        # train setup
        callbacks = list()
        if checkpoints:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    checkpoints,
                    verbose=1,
                    save_weights_only=True
                ),
            )

        def schedule(epoch, decay=0.9):
            return learning_rate * decay**(epoch)
        callbacks.append(keras.callbacks.LearningRateScheduler(schedule))

        if optimizer is None:
            optim = keras.optimizers.Adam(lr=learning_rate)
            # optim = keras.optimizers.SGD(
            #     lr=learning_rate, momentum=0.9, decay=0.0005, nesterov=True
            # )
        else:
            optim = optimizer

        self.model.compile(
            optimizer=optim,
            loss=MultiBoxLoss(
                self.n_classes,
                neg_pos_ratio=neg_pos_ratio
            ).compute_loss
        )
        history = self.model.fit_generator(
            gen.generate(self.preprocesser, True),
            int(gen.train_batches/gen.batch_size),
            epochs=epoch,
            verbose=1,
            callbacks=callbacks,
            validation_data=gen.generate(self.preprocesser, False),
            validation_steps=int(gen.val_batches/gen.batch_size),
            workers=1
        )

        return history

    def save_parameters(self, filepath="./param.json"):
        """
        """
        params = dict(
            n_classes=self.n_classes,
            class_names=self.class_names,
            input_shape=self.input_shape,
            model_type=self.model_type,
            base_net=self.base_net,
            aspect_ratios=self.aspect_ratios,
            scales=self.scales,
            variances=self.variances
        )
        print("Writing parameters into {}.".format(filepath))
        json.dump(params, open(filepath, "w"), indent=4, sort_keys=True)

    def load_parameters(self, filepath):
        """
        """
        print("Loading parameters from {}.".format(filepath))
        params = json.load(open(filepath, "r"))
        self.n_classes = params["n_classes"]
        self.class_names = params["class_names"]
        self.input_shape = params["input_shape"]
        self.model_type = params["model_type"]
        self.base_net = params["base_net"]
        self.aspect_ratios = params["aspect_ratios"]
        self.scales = params["scales"]
        self.variances = params["variances"]

    def detect(self, X, batch_size=1, verbose=0,
               keep_top_k=200, confidence_threshold=0.01,
               do_preprocess=True):
        """
        """
        if do_preprocess:
            inputs = self.preprocesser(X.copy())
        else:
            inputs = X.copy()

        predictions = self.model.predict(inputs,
                                         batch_size=batch_size,
                                         verbose=verbose)
        detections = self.bboxes.detection_out(
            predictions,
            keep_top_k=keep_top_k,
            confidence_threshold=confidence_threshold
        )

        return detections
