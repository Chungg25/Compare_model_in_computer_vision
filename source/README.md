## Running the Models

### **1. Convolutional Neural Networks (CNNs)**

-   **File**: `CNN.ipynb`
-   **Steps**:
    1. Open the notebook in Jupyter or any compatible IDE.
    2. Execute the cells to load data, train the model, and evaluate it.
    3. Output includes accuracy/loss graphs and predictions on the test set.

---

### **2. Mask R-CNN**

-   **File**: `Mask_R_CNN.ipynb`
-   **Reference**: [PyTorch Train Mask R-CNN Tutorial](https://christianjmills.com/posts/pytorch-train-mask-rcnn-tutorial/)
-   **Steps**:
    1. Install required libraries:
        ```bash
        pip install torch torchvision pycocotools matplotlib
        ```
    2. Prepare the dataset:
        - Download the COCO dataset or prepare a custom dataset annotated in COCO format.
        - Update the dataset path in the notebook.
    3. Train the model by running the cells in the notebook.
    4. Evaluate the results on test images to visualize object detection and segmentation.

---

### **3. AutoEncoder**

-   **File**: `Autoencoder.ipynb`
-   **Steps**:
    1. Prepare a dataset such as MNIST.
    2. Open the notebook and execute the cells to train the autoencoder.
    3. Observe the reconstructed images and compare them to the originals.

---

### **4. DCGAN**

-   **File**: `DCGAN.ipynb`
-   **Reference**: [PyTorch DCGAN Faces Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
-   **Steps**:
    1. Install the required libraries:
        ```bash
        pip install torch torchvision matplotlib opencv-python
        ```
    2. Download a dataset (e.g., CelebA or a custom dataset).
    3. Update the dataset path in the notebook.
    4. Train the model by executing the cells in the notebook.
    5. Generated images will be saved in the output directory after each epoch.

---

### **5. Vision Transformer (ViT)**

-   **File**: `Vision_Transformers.ipynb`
-   **Reference**: [Vision Transformer from Scratch](https://github.com/tintn/vision-transformer-from-scratch/blob/main/vision_transformers.ipynb)
-   **Steps**:
    1. Install the required libraries:
        ```bash
        pip install torch torchvision matplotlib
        ```
    2. Download a dataset such as CIFAR-10 or ImageNet.
    3. Open the notebook and configure the dataset path.
    4. Train the Vision Transformer model by running the notebook cells.
    5. Observe the training progress, attention maps, and classification results.

---

## Notes

-   Ensure GPU is enabled for faster training.
-   Adjust hyperparameters such as learning rate and batch size for optimal results.
-   Use appropriate datasets for specific tasks.
