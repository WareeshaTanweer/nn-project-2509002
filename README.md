# Parking Space Occupancy Classification (Custom CNN)

**Course:** BAI13143 Neural Network Basics, Individual Project
**Student:** Wareesha Tanweer (2509002)

A custom convolutional neural network (PyTorch) that classifies a parking space image as **empty** or **occupied**. Training and testing each run in their own Docker container, so the model can be trained in one environment and tested in a completely separate one.

## Dataset

- Source: PKLot dataset by Almeida et al. (2015), downloaded from the [Parking Lot Dataset page on Kaggle](https://www.kaggle.com/datasets/blanderbuss/parking-lot-dataset) (uploaded by Bohdan Fedorak)
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), as stated in the `licence` file included with the dataset
- The full dataset has about 721k files (5.5 GB). This project uses a **balanced subset of 5,000 images** (2,500 empty, 2,500 occupied), created by `src/make_subset.py` and stored in `data/`.
- Preprocessing: images resized to 64x64 and normalised to the range -1 to 1 (mean 0.5, std 0.5).
- Split: 70% train / 15% validation / 15% test (3,500 / 750 / 750 images) using a fixed random seed (42), so the split is identical on every machine.

## Project structure

```
nn-project-2509002/
├── data/                  dataset (mounted into containers, not copied into images)
│   ├── empty/
│   └── occupied/
├── models/                trained models, training histories, preprocessing.json
├── src/
│   ├── dataset.py         loading, preprocessing and train/val/test split
│   ├── model.py           the custom CNN (ParkingCNN)
│   ├── train.py           training script
│   ├── test.py            evaluation on the test set
│   ├── predict.py         prediction on a single new image
│   ├── explore.py, check_data.py, make_subset.py, evaluate.py
├── Dockerfile.train       training image
├── Dockerfile.test        smaller, inference-only test image
├── requirements-train.txt
├── requirements-test.txt
├── occupied_1.jpg         sample image for the prediction demo
└── *.png                  data exploration plots, training curves, confusion matrix
```

The saved model used for the final results is `models/docker_model.pt`.

## Requirements

- Docker Desktop (or Docker Engine) installed and running. CPU only is fine.
- Commands below are for Windows Command Prompt. On Mac/Linux, replace `"%cd%\data` with `"$(pwd)/data` and `"%cd%\models` with `"$(pwd)/models`.

## 1. Train (training container)

```
docker build -f Dockerfile.train -t nn-train .
docker run --rm -v "%cd%\data:/app/data" -v "%cd%\models:/app/models" nn-train
```

This trains the CNN for 15 epochs, prints the training and validation loss and accuracy per epoch, and saves the best model to `models/docker_model.pt` together with `models/preprocessing.json`.

## 2. Test (testing container)

```
docker build -f Dockerfile.test -t nn-test .
docker run --rm -v "%cd%\data:/app/data" -v "%cd%\models:/app/models" nn-test
```

This loads the saved model, runs it on the 750-image test set and prints the metrics. The test image only contains what is needed for inference. It can be run on a different machine from the one used for training: copy the project, `models/` and `data/` folders there, then build and run.

## 3. Predict on a new image

```
docker run --rm --entrypoint python -v "%cd%\models:/app/models" -v "%cd%\occupied_1.jpg:/app/occupied_1.jpg" nn-test src/predict.py occupied_1.jpg
```

Replace `occupied_1.jpg` (in both places) with your own image. Expected output for the sample image:

```
Prediction: occupied (confidence 100.0%)
```

## Results

Test set (750 images, evaluated once):

| Metric | Value |
|---|---|
| Accuracy | 0.996 |
| Precision | 0.9974 |
| Recall | 0.9948 |
| F1-score | 0.9961 |

Confusion matrix (rows = true, columns = predicted, class order: empty, occupied):

```
[[365,   1],
 [  2, 382]]
```

Only 3 of 750 test images were misclassified. The same numbers were obtained on a second, separate machine (different laptop, different Windows version), which confirms the results are reproducible.

## Reproducibility

- Library versions are listed in `requirements-train.txt` and `requirements-test.txt`.
- The train/validation/test split uses a fixed seed (42).
- The dataset and models are mounted into the containers with volumes and are not copied into the images.
- The preprocessing settings are saved in `models/preprocessing.json`.

## Citation

Almeida, P. R. L. de, Oliveira, L. S., Britto Jr., A. S., Silva Jr., E. J., & Koerich, A. L. (2015). PKLot – A robust dataset for parking lot classification. Expert Systems with Applications, 42(11), 4937–4949.
