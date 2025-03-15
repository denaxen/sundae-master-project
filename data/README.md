# Instructions for the data

## [TEMPLATE] Where and how to set up the data

> [!IMPORTANT]
> **TEMPLATE TODO:**
> Update the instructions below to explain how to obtain the data and delete this section.

The template provides the `PROJECT_ROOT/data/` directory as a placeholder for the data used in the project.
This allows the experiment code to always refer to the same path for the data independently of the deployment method
and the user configuration for better reproducibility.
The directory can be accessed in the experiments with `config.data_dir`.
Of course, this doesn't mean that the datasets inside `PROJECT_ROOT/data/` need to be physically in the same directory
as the project.
You can create symlinks to them.
This shifts the data path configuration from the code and config to the installation steps
(which we prefer, as it makes the committed code identical across deployment options).
This is also more convenient than using environment variables to point to individual dataset locations.

Below, you can instruct the users on how to download or link to the data and preprocess it.

When the data is small enough (a few MBs),
you can instruct the users (including you) to download it in the `PROJECT_ROOT/data/` directory.

Otherwise, you can provide hints to them on how to download it (or reuse parts of it) in a separate storage
(likely in a shared storage where some datasets already exist) and then create symlinks to the different parts.
For managed clusters you need to mount different filesystems remember to add this to the deployment scripts
and setup files (e.g. `compose.yaml` for deployment with Docker.)

Here are example instructions:

To setup the `data` directory you can download the data anywhere on your system and then symlink to the data from
the `PROJECT_ROOT/data/` directory.

```bash
# The data set already exist at /absolute_path/to/some-dataset
# FROM the PROJECT_ROOT do
ln -s /absolute-path/to/some-dataset data/some-dataset
# Do this for each dataset root.
# TEMPLATE TODO list all dataset roots (it's better to group them and use the groups accordingly in your code).
```

Be mindful that for the different deployment methods with container engines you will have to mount the filesystems
where the data is stored (E.g. the local deployment option with Docker, and the container deployment on managed clusters)

`TEMPLATE TODO:` For the local deployment option with Docker you would edit the `../installation/docker-*/compose.yaml`
file for the local deployment option with Docker,
for the managed clusters you would edit the flags of the cluster client (`runai`, `srun`, etc.).
Avoid nested mounts.
It's better to mount the whole "scratch" filesystem and let the symlinks handle the rest.

## Description of the data

### text8 Dataset
The text8 dataset is a preprocessed version of the English Wikipedia text, consisting of the first 100 million characters. It has been cleaned to contain only lowercase letters a-z and spaces.

### C4 Dataset (Colossal Clean Common Crawl)
The C4 dataset, or "Colossal Clean Common Crawl", is a large dataset created by cleaning Common Crawl web data. It consists of hundreds of gigabytes of English text scraped from the web, filtered to remove incomplete sentences, boilerplate text like menus and navigation, offensive content, and non-English text.

## Instructions to obtain the data

### text8 Dataset
The text8 dataset can be downloaded from:
```bash
wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip -d data/
```

### C4 Dataset
The full C4 dataset is very large (several hundred GB). There are a few ways to access it:

#### Option 1: Using Hugging Face Datasets
You can use the Hugging Face Datasets library to stream the dataset:

```bash
pip install datasets
```

Then create a script to download and prepare the data:

```python
from datasets import load_dataset
import json
import os

# Create output directory
os.makedirs('data/c4', exist_ok=True)

# Load a small subset for validation (adjust split and streaming as needed)
dataset = load_dataset('c4', 'en', split='validation', streaming=True)

# Write a sample of validation data to a jsonl file
with open('data/c4/validation.jsonl', 'w') as f:
    for i, example in enumerate(dataset):
        if i >= 10000:  # Adjust sample size as needed
            break
        f.write(json.dumps(example) + '\n')

# Load a subset of training data
dataset = load_dataset('c4', 'en', split='train', streaming=True)

# Write a sample of training data to a jsonl file
with open('data/c4/train.jsonl', 'w') as f:
    for i, example in enumerate(dataset):
        if i >= 100000:  # Adjust sample size as needed
            break
        f.write(json.dumps(example) + '\n')
```

#### Option 2: TensorFlow Datasets
Alternatively, you can use TensorFlow Datasets:

```bash
pip install tensorflow tensorflow-datasets
```

```python
import tensorflow_datasets as tfds
import json
import os

os.makedirs('data/c4', exist_ok=True)

# Load validation data
dataset = tfds.load('c4/en:3.0.1', split='validation')

# Write validation data to jsonl
with open('data/c4/validation.jsonl', 'w') as f:
    for i, example in enumerate(dataset):
        if i >= 10000:  # Adjust as needed
            break
        item = {
            'text': example['text'].numpy().decode('utf-8'),
            'url': example['url'].numpy().decode('utf-8')
        }
        f.write(json.dumps(item) + '\n')

# Load training data
dataset = tfds.load('c4/en:3.0.1', split='train')

# Write training data to jsonl
with open('data/c4/train.jsonl', 'w') as f:
    for i, example in enumerate(dataset):
        if i >= 100000:  # Adjust as needed
            break
        item = {
            'text': example['text'].numpy().decode('utf-8'),
            'url': example['url'].numpy().decode('utf-8')
        }
        f.write(json.dumps(item) + '\n')
```

#### Option 3: Direct Download from TensorFlow
For the full dataset, you can download it directly from the TensorFlow website:
https://www.tensorflow.org/datasets/catalog/c4

## Instructions to process the data
The C4 dataset is pre-processed and ready to use. Just make sure the data is in the expected format:
- JSONL files with each line containing a JSON object
- Each JSON object should have a 'text' field

When running experiments with C4, use the `c4.toml` configuration file:
```bash
python main.py --cfg-path config/c4.toml
```

If you're using only a subset of C4 for testing, you can uncomment and set the `max_examples` parameter in the config file.
