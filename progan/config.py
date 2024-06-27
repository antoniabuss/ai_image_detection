import torch
from math import log2

START_TRAIN_AT_IMG_SIZE = 4
DATASET = 'dataset'
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-4
BATCH_SIZES = [16, 16, 16, 8, 8, 8, 8, 4, 2]
CHANNELS_IMG = 3
Z_DIM = 32  # should be 512 in original paper
IN_CHANNELS = 32  # should be 512 in original paper
LAMBDA_GP = 10
# if not good set higher
PROGRESSIVE_EPOCHS = [10] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4