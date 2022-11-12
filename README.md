# Install Required Libraries
 * Run the command `pip install -r requirements.txt`

# Retrain the Model With New Images
 * In the folder `train`, there are folders of the classes of which to be classified. Each folder contains images of each class (example, `apples` folder contains images of apples).
 * The images must all be JPG.
 * After putting all images, run the command `python train.py`. After training, it will save a model that can be used later on for testing.

# Test the Model
* In the file `test.py` at line 13, there is the image of the file that will be tested. This file can be replaced with an image on the internet. Again, all images must be JPG.
* To test, run the command `python test.py`.