# Overview
This project is an Keras implementation of U-net architecture that is typically used for image segmentation.
I attempted to use it to generate masks that would remove artifacts from compressed jpeg images. The implementation allows for restoration of image of any size.  
Research conclusion is that the model has characteristics of an averaging filter. By slightly blurring the image it hides the edges between image segments that jpeg algorithm creates. 
While human perception of image can improve, original data cannot be restored.
# Content
1. model.py - contains Keras implementation of U-net architecture
2. augmentation.py - data loading and augmentation
3. data.py - creates data generators
4. restore.py - functions responsible for processing and restoring compressed image of any size
5. sprawozdanie/sprawozdanie.pdf - report in Polish including insights on related technologies and results of the project
