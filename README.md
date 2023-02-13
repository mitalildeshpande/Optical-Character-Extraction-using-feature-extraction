# Optical-Character-Extraction-using-feature-extraction

This project is about implementing an optical character recognition system using one of the inbuilt feature extraction and transformation libraries in openCV. I experimented with connected component and matching algorithms where the goal was to detect and recognize various characters.
The first input was be a collection with an arbitrary number of target characters as individual image files which each represented a character to recognize in the from of a ”template”.
The second input was a gray scale test image containing characters to recognize. The input image had characters that are darker than the background. The background was defined by the color that is touching the boundary of the image. All characters were separated from each other by at least one background pixel but may have different gray levels.

The OCR system will contain three parts, Enrollment, Detection and Recognition.

For the enrollment process, we process a set of target characters from the provided directory and generate features for each suitable for classification/recognition in Recognition. We store these features in any way you want to in an intermediate file, that is read in by the recognizer. The file we store should NOT be the same as an image file. The reason for the intermediate file is so we do not have to run enrollment every time you want to run detection and recognition on a new image.
For the Detection process, we will need to use connected component labeling to detect various candidate characters in the image. We should identify ALL possible candidates in the test image even if they do not appear in the list of enrolled characters. The characters can be between 1/2 and 2x the size of the enrolled images. Once we have detected the character positions, we are free to generate any features or resize those areas of the image in any way you want in preparation for recognition. The detection results should be stored for output with the recognition results in recognition, and should be in the original image coordinates.
For the recognition process, we take each of the candidate characters detected from previous part, and our features for each of the enrolled characters, we are required to implement a recognition or matching function that determines the identity of the character if it is in the enrollment set or UNKNOWN if it is not.

## As per the output, we generate an output file ‘results.json’ which is a list with each entry as {“bbox” : [x (integer), y (integer), w (integer), h (integer)],“name”: (string)}.

An evaluation script is provided in the repository to evaluated the performance of the OCR using confusion matrix. It also provides an F1 score for the implementation to give a good idea of the performance.
