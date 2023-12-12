# ImageClustering

Input Data (provided under Test) consists of 10,740 images of handwritten digits (0-9). The images were scanned and scaled into 28x28 pixels. For every digit, each pixel can be represented as an integer in the range [0, 255] where 0 corresponds to the pixel being completely white, and 255 corresponds to the pixel being completely black. This gives us a 28x28 matrix of integers for each digit. We can then flatten each matrix into a 1x784 vector. No labels are provided.

**Format of the input data:**
Each row is a record (image), which contains 784 comma-delimited integers.