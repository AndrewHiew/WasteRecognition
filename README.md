# A University AI Research Project
An AI Project made for my University Project.

## Setting up Environment

Run the following command to setup a python environment to run this project.
Make sure you are in the correct directory before executing this command.
```
cd YOUR_DIRECTORY
pip install -r requirements.txt
```

## File Description
The following files are used to train the Model.
- vgg19_model.py
- restnet50_model.py
- svm_model.py
After training, the quantised model should be under this directory. models/YOUR_MODEL

## Testing the Model
Run the following command to initiate the UI.
```
streamlit run ui.py
```
Use the camera or simply upload an image, the model will classify the image accordingly.
