# Brain tumour MRI classifier (GDG DevFest 2025 workshop)

This is the notebook from the workshop I ran at GDG DevFest 2025 at Bletchley Park. Attendees build an image classifier in Google Colab using transfer learning on MobileNetV3Large with TensorFlow and Keras. It sorts brain MRI scans into the three classes in the dataset's brain folder: glioma, meningioma and a class labelled `brain_tumor`. The notebook then looks at the results with a confusion matrix, sample predictions and a Grad-CAM heatmap.

This project is not a medical tool. It was written for teaching and must not be used to diagnose anyone or to make any healthcare decision.

## How it works

1. Download the Kaggle dataset `obulisainaren/multi-cancer` (about 8.6 GB) with `kagglehub`.
2. Load only the `Brain Cancer` folder with `image_dataset_from_directory`: 15,000 images in three classes (`brain_glioma`, `brain_menin`, `brain_tumor`), resized to 224 by 224, one-hot labels, batch size 16. An 80/20 split with seed 123 gives 12,000 training and 3,000 validation images.
3. Load MobileNetV3Large with ImageNet weights, drop its top layer and freeze it.
4. Add global average pooling, dense layers of 1024, 512 and 128 units (ReLU) and a three-way softmax output.
5. Train for 2 epochs with Adam (learning rate 0.0001) and categorical cross-entropy, then save the model as `brain_cancer_model.keras`.
6. Plot the class balance, the training curves, a confusion matrix and nine sample predictions.
7. Run Grad-CAM on the `conv_1_bn` layer to show which parts of a scan drove a prediction.

## Results

These are from the saved Colab run, measured on the 3,000 validation images.

| Measure | Value |
|---|---|
| Validation accuracy after 2 epochs | 97.7% |
| Validation loss | 0.075 |
| Training accuracy, epoch 2 | 97.6% |

Confusion matrix on the validation set (rows are true labels):

| | Predicted glioma | Predicted meningioma | Predicted brain_tumor |
|---|---|---|---|
| Glioma | 979 | 38 | 0 |
| Meningioma | 0 | 981 | 6 |
| brain_tumor | 0 | 24 | 972 |

## Repository layout

| Path | What it is |
|---|---|
| `code.ipynb` | The workshop notebook, with outputs from a Colab run |

## Following along

[Open in Colab](https://colab.research.google.com/github/lewis-shearer/GDG_Bletchley_Park_Workshop/blob/main/code.ipynb)

You need a Google account. You can also open it from Colab with File > Open notebook > GitHub and the repository link https://github.com/lewis-shearer/GDG_Bletchley_Park_Workshop.

Turn on a GPU before running anything: Runtime > Change runtime type, choose a GPU (a T4 is fine) and click Save. To check it worked, run this in a new cell and make sure a GPU is listed:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

Then run the cells from top to bottom with Shift+Enter. The dataset download takes a couple of minutes, and each training epoch took 30 to 45 seconds on the GPU in my run. The notebook uses TensorFlow, Keras, kagglehub, NumPy, Matplotlib, seaborn, scikit-learn and OpenCV, which are all available in Colab.

## Limitations

The same 3,000 validation images are used to monitor training and to report the final accuracy, so there is no separate held-out test set. The model is trained for only 2 epochs to fit the workshop time slot (a code comment still says 10). The notebook cannot tell a healthy scan from a tumour, because every class in this part of the dataset is a type of tumour.

Presented by Lewis Shearer ([LinkedIn](https://www.linkedin.com/in/lewis-shearer/)).
