import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Ensure logs directory exists
import os
os.makedirs("logs", exist_ok=True)

# Setup logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AeroEye")

# Base directory path resolution for robust absolute path construction
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset directory locations
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
YOLO_DATA_DIR = os.path.join(BASE_DIR, 'dataset/yolo_format')

# Split-specific directories for classification pipeline
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Model output directory
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Results and visualization save directory
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------
# Pipeline Hyperparameters
# ----------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Targets
CLASSES = ['bird', 'drone']
