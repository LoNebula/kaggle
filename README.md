# kaggle

# Build and Run the Container
Run the following command to build the image from the Dockerfile and start the container in the background according to the settings in docker-compose.yml.
```bash
docker compose up --build -d
```

# Enter the Container
To get a shell inside the running container, execute the following command.
```bash
docker compose exec app /bin/bash
```

# Verify GPU Access
After entering the container, start Python and run the following code to check if PyTorch is correctly recognizing the GPU.
```python
import torch
print(torch.cuda.is_available())
```

```md
# kaggle


# Build and Run the Container
Run the following command to build the image from the Dockerfile and start the container in the background according to the settings in docker-compose.yml.
```bash
docker compose up --build -d
```


# Enter the Container
To get a shell inside the running container, execute the following command.
```bash
docker compose exec app /bin/bash
```


# Verify GPU Access
After entering the container, start Python and run the following code to check if PyTorch is correctly recognizing the GPU.
```python
import torch
print(torch.cuda.is_available())
```


# Kaggle Setup in VSCode

## 1. Install Kaggle API inside the Container

Inside the container, install the Kaggle API:

```bash
pip install kaggle
```

Verify the installation:

```bash
kaggle --version
```


## 2. Authenticate the Kaggle API

First, obtain your API token from Kaggle:

1. Open https://www.kaggle.com/account in your browser.
2. In the “API” section, click “Create New API Token”.
3. A file named `kaggle.json` will be downloaded.

Make this `kaggle.json` visible to the container and place it where the Kaggle CLI expects it (inside the container home directory):

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Test authentication:

```bash
kaggle competitions list | head
```


## 3. (Optional) Download Competition Data Locally

If you *do* want to download data into the container, use the competition slug (e.g., `spaceship-titanic`):

```bash
# List available files for the competition
kaggle competitions files -c spaceship-titanic

# Download all files to the current directory
kaggle competitions download -c spaceship-titanic

# Download into a specific folder (e.g., ./data)
kaggle competitions download -c spaceship-titanic -p ./data
```

If you want to save storage, you can skip this step and use the Kaggle Notebook + VSCode workflow described next.


## 4. Use Kaggle Notebook + VSCode (No Local Dataset)

To avoid downloading datasets locally, use Kaggle’s hosted environment for data and compute, and connect to it from VSCode.

### 4.1 Start a Kaggle Notebook and Jupyter Server

1. In your browser, go to Kaggle → **Notebooks** and create a new notebook or open an existing one for the target competition (e.g., Spaceship Titanic).
2. In the notebook UI, click **Add data** and attach the competition dataset (e.g., `spaceship-titanic`) and any other required datasets.
3. In the top-right corner, open the **Run** menu and choose **Kaggle Jupyter Server**, enabling GPU if needed.
4. Wait for the Jupyter Server panel to show that the server is running.

### 4.2 Get the VSCode-Compatible Jupyter URL

1. In the Jupyter Server panel on the right side of the Kaggle Notebook, locate the **VSCode compatible URL**.
2. Copy this URL (it looks like `http://...:8888/?token=...`).

### 4.3 Connect from VSCode to the Kaggle Jupyter Server

1. In VSCode (inside the container or on your host), open your local `.ipynb` file (e.g., `spaceship-titanic_2.ipynb`).
2. Click **Select Kernel** in the top-right of the notebook editor.
3. Choose **Existing Jupyter Server** (or similar wording).
4. When prompted for a URL, paste the **VSCode compatible URL** copied from Kaggle.
5. After connection, run the following cell to confirm you are using the Kaggle GPU:

   ```python
   import torch
   print(torch.cuda.is_available())
   ```

6. Load data directly from Kaggle’s filesystem:

   ```python
   import pandas as pd

   train = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
   test = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")
   ```

In this setup, datasets stay on Kaggle’s side; your local/container storage is not used for the competition data.


## 5. Push a Notebook to Kaggle via the API

You can also edit a notebook locally in VSCode and push it to Kaggle using the Kaggle API.

### 5.1 Create `kernel-metadata.json`

In your project directory (e.g., `/workspace/kaggle/spaceship-titanic`), initialize kernel metadata:

```bash
kaggle kernels init -p .
```

Edit the generated `kernel-metadata.json` as follows (example):

```json
{
  "id": "shogomiyawaki/spaceship-titanic-2",
  "title": "Spaceship Titanic 2",
  "code_file": "spaceship-titanic_2.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_tpu": "false",
  "enable_internet": "true",
  "dataset_sources": [],
  "competition_sources": ["spaceship-titanic"],
  "kernel_sources": [],
  "model_sources": []
}
```

Key fields:

- `id`: `username/kernel-slug`, unique per notebook.
- `title`: Human-readable title shown on Kaggle.
- `code_file`: The `.ipynb` file you want to push.
- `competition_sources`: List of competition slugs (e.g., `"spaceship-titanic"`).


### 5.2 Push the Notebook

Ensure `kernel-metadata.json` and your `.ipynb` file are in the same directory, then run:

```bash
kaggle kernels push -p .
```

- On the first push, a new notebook is created on Kaggle.
- Subsequent pushes create new versions of the same notebook (same `id`).
- The command output will show the URL of the notebook, for example:

  `https://www.kaggle.com/code/shogomiyawaki/spaceship-titanic-2`


## 6. Typical Workflow

1. Start the Docker container and open the project in VSCode (Remote Containers or similar).
2. Inside the container:
   - Install and authenticate the Kaggle API.
3. Choose one of:
   - **Kaggle Notebook + VSCode connection**: edit and run notebooks in VSCode while data and compute stay on Kaggle.
   - **Local download (optional)**: use `kaggle competitions download` for small datasets if local storage usage is acceptable.
4. Edit your `.ipynb` in VSCode.
5. Use `kernel-metadata.json` and `kaggle kernels push -p .` to publish/update the notebook on Kaggle when you want to share or get medals.
```