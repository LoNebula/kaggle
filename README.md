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


# Kaggle Setup in VSCode (Generic Workflow)

This section describes a generic workflow to use Kaggle with VSCode and Docker for any competition or dataset, not tied to a specific project.


## 1. Install Kaggle API Inside the Container

Inside the container, install the Kaggle CLI:

```bash
pip install kaggle
```

Verify the installation:

```bash
kaggle --version
```


## 2. Authenticate the Kaggle API

1. Log in to Kaggle in your browser and open your account page:  
   https://www.kaggle.com/account
2. In the **API** section, click **Create New API Token**.
3. A file named `kaggle.json` will be downloaded.

Place `kaggle.json` where the CLI expects it inside the container:

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Test authentication:

```bash
kaggle competitions list | head
```


## 3. Download Data with Kaggle API (Optional)

If you are okay with storing data locally (inside the container), you can download competition data or general datasets.

### 3.1 Competitions

Let `<competition-slug>` be the competition identifier in the URL, e.g.:

- `https://www.kaggle.com/competitions/titanic` → slug: `titanic`
- `https://www.kaggle.com/competitions/spaceship-titanic` → slug: `spaceship-titanic` [web:123]

List files:

```bash
kaggle competitions files -c <competition-slug>
```

Download all files to the current directory:

```bash
kaggle competitions download -c <competition-slug>
```

Download into a specific folder:

```bash
kaggle competitions download -c <competition-slug> -p ./data
```

Download a specific file:

```bash
kaggle competitions download -c <competition-slug> -f train.csv -p ./data
```


### 3.2 Datasets

For general datasets, use the `owner/dataset-slug` form. You can copy the exact command from the “Copy API command” button on the Kaggle dataset page. [web:121][web:118]

```bash
kaggle datasets download -d <owner>/<dataset-slug> -p ./data
```


## 4. Use Kaggle Notebooks + VSCode (No Local Data)

To avoid local storage usage and still enjoy VSCode, connect VSCode to a running Kaggle Jupyter server.

### 4.1 Start a Kaggle Notebook and Jupyter Server

1. In a browser, go to Kaggle → **Notebooks** and create/open a notebook.
2. Click **Add data** in the notebook and attach any required competition/datasets (this will mount them under `/kaggle/input/...`). [web:49][web:122]
3. In the top-right, open the **Run** menu and select **Kaggle Jupyter Server**, enabling GPU if needed.
4. Wait until the Jupyter Server panel shows the server is running. [web:49][web:112]


### 4.2 Get the VSCode-Compatible URL

1. In the Jupyter Server panel (right side of the notebook), find the **VSCode compatible URL**.
2. Copy that URL (it looks like `http://<host>:8888/?token=...`). [web:28][web:113]


### 4.3 Connect from VSCode

1. In VSCode (inside or outside the container), open any `.ipynb` you want to work on.
2. Click **Select Kernel** in the top-right of the notebook editor.
3. Choose **Existing Jupyter Server** (or similar), and when prompted, paste the VSCode-compatible URL from Kaggle. [web:28][web:26]
4. After connecting, verify that the remote environment is active:

   ```python
   import torch
   print(torch.cuda.is_available())
   ```

5. Load data from Kaggle’s mounted paths (generic example):

   ```python
   import pandas as pd

   # Example for a competition:
   df = pd.read_csv("/kaggle/input/competitions/<competition-slug>/train.csv")

   # Example for an attached dataset:
   # df = pd.read_csv("/kaggle/input/<dataset-folder>/data.csv")
   ```

In this mode, data lives on Kaggle; your local/container storage is not used for datasets. [web:105][web:49]


## 5. Push Notebooks to Kaggle via Kaggle API

You can keep notebooks in Git and push them to Kaggle programmatically using `kernel-metadata.json`. This works for any competition or dataset.

### 5.1 Initialize Kernel Metadata

In your project directory (per competition or per notebook), run:

```bash
kaggle kernels init -p .
```

This generates a `kernel-metadata.json` template. Edit it to match your project:

```json
{
  "id": "your-username/your-kernel-slug",
  "title": "Your Kernel Title",
  "code_file": "your-notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_tpu": "false",
  "enable_internet": "false",
  "dataset_sources": [],
  "competition_sources": ["<competition-slug>"],
  "kernel_sources": [],
  "model_sources": []
}
```

Notes: [web:122][web:125]

- `id` must be unique per notebook: `username/slug`.
- `title` is the human-readable title on Kaggle.
- `code_file` is the `.ipynb` file to push.
- `competition_sources`: list of competition slugs whose data you want mounted.
- `dataset_sources`: list of dataset identifiers (`owner/dataset-slug`) if you want them mounted.


### 5.2 Push the Notebook

With `kernel-metadata.json` and the notebook in the same directory:

```bash
kaggle kernels push -p .
```

- On first push, a new notebook is created on Kaggle.
- Subsequent pushes create new versions of the same notebook (same `id`). [web:118]
- The CLI output will print the Kaggle URL of the notebook (you can open it to check status and run it in the browser).


## 6. Directory Layout Example

A generic layout for multiple competitions:

```text
kaggle-projects/
├── <competition-a>/
│   ├── kernel-metadata.json
│   └── main-a.ipynb
├── <competition-b>/
│   ├── kernel-metadata.json
│   └── main-b.ipynb
└── README.md
```

Each competition folder has its own metadata and notebook(s). [web:122]


## 7. Typical End-to-End Workflow

1. Start the Docker container and open the project in VSCode.
2. Inside the container:
   - Install and authenticate the Kaggle API.
3. Choose how to access data:
   - **Remote only**: Use Kaggle Notebook + VSCode connection (no local data).
   - **Local (optional)**: Use `kaggle competitions download` / `kaggle datasets download` for small or temporary data.
4. Develop your notebook in VSCode.
5. When ready to publish or version on Kaggle, configure `kernel-metadata.json` and run:

   ```bash
   kaggle kernels push -p .
   ```

6. Open the printed Kaggle URL to view, run, and share the notebook (and eventually earn medals).
```