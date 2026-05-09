import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

def read_file(path):
    with open(os.path.join(script_dir, path), 'r', encoding='utf-8') as f:
        return f.read()

my_model_code = read_file('network/my_model.py')
train_rafdb_code = read_file('train_rafdb.py')
train_affectnet_code = read_file('train_affectnet-7.py')
evaluation_code = read_file('evaluation.py')

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Improved CMNet Training on Google Colab\n",
            "This notebook applies our architectural improvements (ResNet50 backbone, adaptive fusion) and training optimizations (AdamW, Cosine Annealing, Label Smoothing, RandAugment) to the CMNet model."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!git clone https://github.com/hellloxiaotian/CMNet.git\n",
            "%cd CMNet"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "os.makedirs('data', exist_ok=True)\n",
            "os.makedirs('models', exist_ok=True)\n",
            "os.makedirs('experiment/visual/confusion_matrix', exist_ok=True)\n",
            "os.makedirs('experiment/rafdb', exist_ok=True)\n",
            "os.makedirs('checkpoint', exist_ok=True)\n",
            "os.makedirs('log', exist_ok=True)\n",
            "\n",
            "!pip install -U gdown\n",
            "# Download datasets\n",
            "!gdown --folder https://drive.google.com/drive/folders/1OjFbQ7ykV5x96MrAMySHwMpaJKzFgGSe -O data_tmp\n",
            "# Download models\n",
            "!gdown --folder https://drive.google.com/drive/folders/1kPUbCKoDKesxbpubPTxedD3DWsGGJ1qj -O models_tmp\n",
            "\n",
            "# Move contents to appropriate folders\n",
            "!mv data_tmp/* data/ || true\n",
            "!mv models_tmp/* models/ || true"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%writefile network/my_model.py\n"
        ] + [line + '\n' for line in my_model_code.split('\n')]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%writefile train_rafdb.py\n"
        ] + [line + '\n' for line in train_rafdb_code.split('\n')]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%writefile train_affectnet-7.py\n"
        ] + [line + '\n' for line in train_affectnet_code.split('\n')]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%writefile evaluation.py\n"
        ] + [line + '\n' for line in evaluation_code.split('\n')]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install required dependencies\n",
            "!pip install pandas matplotlib tqdm torchvision"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Run Training on RAF-DB\n",
            "!python train_rafdb.py"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import glob\n",
            "import shutil\n",
            "# Find the best checkpoint and set it up for evaluation.py\n",
            "best_ckpt = glob.glob('checkpoint/*_best.pth')\n",
            "if best_ckpt:\n",
            "    print(f'Found best checkpoint: {best_ckpt[0]}')\n",
            "    shutil.copy(best_ckpt[0], 'experiment/rafdb/rafdb.pth')\n",
            "else:\n",
            "    print('No checkpoint found!')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Run Evaluation on RAF-DB\n",
            "!python evaluation.py"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import Image, display\n",
            "import glob\n",
            "\n",
            "# Display Training Curves\n",
            "curve_images = glob.glob('log/*.png')\n",
            "if curve_images:\n",
            "    print('Training & Validation Curves:')\n",
            "    display(Image(filename=curve_images[-1]))\n",
            "\n",
            "# Display Confusion Matrix\n",
            "cm_images = glob.glob('experiment/visual/confusion_matrix/*.png')\n",
            "if cm_images:\n",
            "    print('\\nConfusion Matrix:')\n",
            "    display(Image(filename=cm_images[-1]))\n"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {
            "name": "Colab_CMNet_Training.ipynb",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(os.path.join(script_dir, 'Colab_CMNet_Training.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook created successfully.")
