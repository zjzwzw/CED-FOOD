from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

_CC = _C
def add_food_config(cfg):
    _CC = cfg
    # ----------- Backbone ----------- #
    _CC.MODEL.BACKBONE.FREEZE = False
    _CC.MODEL.BACKBONE.FREEZE_AT = 3

    # ------------- RPN -------------- #
    _CC.MODEL.RPN.FREEZE = False
    _CC.MODEL.RPN.ENABLE_DECOUPLE = False
    _CC.MODEL.RPN.BACKWARD_SCALE = 1.0

    # ------------- ROI -------------- #
    _CC.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"
    _CC.MODEL.ROI_HEADS.FREEZE_FEAT = False
    _CC.MODEL.ROI_HEADS.ENABLE_DECOUPLE = False
    _CC.MODEL.ROI_HEADS.BACKWARD_SCALE = 1.0
    _CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
    _CC.MODEL.ROI_HEADS.CLS_DROPOUT = False
    _CC.MODEL.ROI_HEADS.DROPOUT_RATIO = 0.5
    _CC.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES = 40
    _CC.MODEL.ROI_HEADS.NUM_BASE_CLASSES = 20

    # register RoI output layer
    _CC.MODEL.ROI_BOX_HEAD.OUTPUT_LAYERS = "FastRCNNOutputLayers"

    _CC.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7  # for faster
    # scale for cosine classifier
    _CC.MODEL.ROI_HEADS.COSINE_SCALE = 20
    # thresh for visualization results.
    _CC.MODEL.ROI_HEADS.VIS_IOU_THRESH = 1.0

    # ------------- TEST ------------- #
    _CC.TEST.PCB_ENABLE = False
    _CC.TEST.PCB_MODELTYPE = 'resnet'             # res-like
    _CC.TEST.PCB_MODELPATH = ""
    _CC.TEST.PCB_ALPHA = 0.50
    _CC.TEST.PCB_UPPER = 1.0
    _CC.TEST.PCB_LOWER = 0.05
    _CC.TEST.SAVE_FIG = False
    _CC.TEST.SCORE_THREHOLD = 0.5
    _CC.TEST.IMG_OUTPUT_PATH = ""
    _CC.TEST.SAVE_FEATURE_MAP = False

    # ------------ Other ------------- #
    _CC.SOLVER.WEIGHT_DECAY = 5e-5
    _CC.MUTE_HEADER = True
    # _CC.SOLVER.OPTIMIZER = 'SGD'

    # unknown probability loss
    _CC.UPLOSS = CN()
    _CC.UPLOSS.ENABLE_UPLOSS = False
    _CC.UPLOSS.START_ITER = 0  # usually the same as warmup iter
    _CC.UPLOSS.SAMPLING_METRIC = "min_score"
    _CC.UPLOSS.TOPK = 3
    _CC.UPLOSS.TOPM = 1
    _CC.UPLOSS.SAMPLING_RATIO = 1
    _CC.UPLOSS.ALPHA = 1.0
    _CC.UPLOSS.WEIGHT = 0.5

    # instance contrastive loss
    _CC.ICLOSS = CN()
    _CC.ICLOSS.OUT_DIM = 128
    _CC.ICLOSS.QUEUE_SIZE = 256
    _CC.ICLOSS.IN_QUEUE_SIZE = 16
    _CC.ICLOSS.BATCH_IOU_THRESH = 0.5
    _CC.ICLOSS.QUEUE_IOU_THRESH = 0.7
    _CC.ICLOSS.TEMPERATURE = 0.1
    _CC.ICLOSS.WEIGHT = 0.1

    # swin transformer
    _CC.MODEL.SWINT = CN()
    _CC.MODEL.SWINT.EMBED_DIM = 96
    _CC.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    _CC.MODEL.SWINT.DEPTHS = [2, 2, 6, 2]
    _CC.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24]
    _CC.MODEL.SWINT.WINDOW_SIZE = 7
    _CC.MODEL.SWINT.MLP_RATIO = 4
    _CC.MODEL.SWINT.DROP_PATH_RATE = 0.2
    _CC.MODEL.SWINT.APE = False
    _CC.MODEL.SWINT.IN_CHANNELS = 768
    _CC.MODEL.BACKBONE.FREEZE_AT = -1
    _CC.MODEL.FPN.TOP_LEVELS = 2
    _C.MODEL.RPN.CONV_DIMS = [-1]

    # ---------------------------------------------------------------------------- #
    # CLIP options
    # ---------------------------------------------------------------------------- #
    _CC.MODEL.CLIP = CN()

    _CC.MODEL.CLIP.CONCEPT_TXT = ""
    _CC.MODEL.CLIP.BASE_TRAIN = True

    _CC.MODEL.CLIP.CROP_REGION_TYPE = ""  # options: "GT", "RPN"
    _CC.MODEL.CLIP.BB_RPN_WEIGHTS = None  # the weights of pretrained MaskRCNN
    _CC.MODEL.CLIP.IMS_PER_BATCH_TEST = 8  # the #images during inference per batch

    _CC.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER = False  # if True, use the CLIP text embedding as the classifier's weights
    _CC.MODEL.CLIP.TEXT_EMB_PATH = None  # "/mnt/output_storage/trained_models/lvis_cls_emb/lvis_1203_cls_emb.pth"
    _CC.MODEL.CLIP.OFFLINE_RPN_CONFIG = None  # option: all configs of pretrained RPN
    _CC.MODEL.CLIP.NO_BOX_DELTA = False  # if True, during inference, no box delta will be applied to region proposals

    _CC.MODEL.CLIP.BG_CLS_LOSS_WEIGHT = None  # if not None, it is the loss weight for bg regions
    _CC.MODEL.CLIP.ONLY_SAMPLE_FG_PROPOSALS = False  # if True, during training, ignore all bg proposals and only sample fg proposals
    _CC.MODEL.CLIP.MULTIPLY_RPN_SCORE = False  # if True, during inference, multiply RPN scores with classification scores
    _CC.MODEL.CLIP.VIS = False  # if True, when visualizing the object scores, we convert them to the scores before multiplying RPN scores

    _CC.MODEL.CLIP.OPENSET_TEST_NUM_CLASSES = None  # if an integer, it is #all_cls in test
    _CC.MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH = None  # if not None, enables the openset/zero-shot training, the category embeddings during test

    _CC.MODEL.CLIP.CLSS_TEMP = 0.01  # normalization + dot product + temperature
    _CC.MODEL.CLIP.RUN_CVPR_OVR = False  # if True, train CVPR OVR model with their text embeddings
    _CC.MODEL.CLIP.FOCAL_SCALED_LOSS = None  # if not None (float value for gamma), apply focal loss scaling idea to standard cross-entropy loss

    _CC.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH = None  # the threshold of NMS in offline RPN
    _CC.MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST = None  # the number of region proposals from offline RPN
    _CC.MODEL.CLIP.PRETRAIN_IMG_TXT_LEVEL = True  # if True, pretrain model using image-text level matching
    _CC.MODEL.CLIP.PRETRAIN_RPN_REGIONS = None  # if not None, the number of RPN regions per image during pretraining
    _CC.MODEL.CLIP.PRETRAIN_SAMPLE_REGIONS = None  # if not None, the number of regions per image during pretraining after sampling, to avoid overfitting
    _CC.MODEL.CLIP.GATHER_GPUS = False  # if True, gather tensors across GPUS to increase batch size
    _CC.MODEL.CLIP.GRID_REGIONS = False  # if True, use grid boxes to extract grid features, instead of object proposals
    _CC.MODEL.CLIP.CONCEPT_POOL_EMB = None  # if not None, it provides the file path of embs of concept pool and thus enables region-concept matching
    _CC.MODEL.CLIP.CONCEPT_THRES = None  # if not None, the threshold to filter out the regions with low matching score with concept embs, dependent on temp (default: 0.01)

    _CC.MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED = False  # if True, use large-scale jittering (LSJ) pretrained RPN
    _CC.MODEL.CLIP.TEACHER_RESNETS_DEPTH = 50  # the type of visual encoder of teacher model, sucha as ResNet 50, 101, 200 (a flag for 50x4)
    _CC.MODEL.CLIP.TEACHER_CONCEPT_POOL_EMB = None  # if not None, it uses the same concept embedding as student model; otherwise, uses a seperate embedding of teacher model
    _CC.MODEL.CLIP.TEACHER_POOLER_RESOLUTION = 14  # RoIpooling resolution of teacher model

    _CC.MODEL.CLIP.TEXT_EMB_DIM = 1024  # the dimension of precomputed class embeddings
    _CC.INPUT_DIR = "./datasets/custom_images"  # the folder that includes the images for region feature extraction
    _CC.MODEL.CLIP.GET_CONCEPT_EMB = False  # if True (extract concept embedding), a language encoder will be created

    # Use soft NMS instead of standard NMS if set to True
    _CC.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    # See soft NMS paper for definition of these options
    _CC.MODEL.ROI_HEADS.SOFT_NMS_METHOD = "gaussian"  # "linear"
    _CC.MODEL.ROI_HEADS.SOFT_NMS_SIGMA = 0.5
    # For the linear_threshold we use NMS_THRESH_TEST
    _CC.MODEL.ROI_HEADS.SOFT_NMS_PRUNE = 0.001

