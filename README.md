# VLSI-Fault-Detection

## Abstract
Physical Failure Analysis (PFA) is an important process that identifies layout designs contributing to failures in Large-Scale Integration (LSI) circuits. Due to their efficiency and cost effectiveness, ***Convolutional Neural Networks (CNNs)*** have emerged as powerful tools for analyzing LSI layouts. However, the application of CNNs to root cause analysis faces significant challenges. Information on failures is often sparse, and understanding the impact of layout deigns on defects needs information from large regions having a multitude of geometries. These factors are sources of hindrance in successful training of CNN models with high accuracy in layout segment classification.<br />

This project presents a CNN-based algorithm for layout segment classification. This is an attempt to know the way layouts contribute to failure. The proposed approach relies on the use of several segment images of LSI layouts as inputs to the CNN models for segment classification as either "risk" or "non-risk". The 9-layer model reached an ***accuracy of 99%***. This shows the promise of deep learning techniques in improving layout designs and reducing failure risks in the manufacturing process of LSI circuits.<br />

## Problem Statement
Testing is one of the most important stages in chip manufacturing and most cost-intensive, mainly because it guarantees the reliability and performance of every chip shipped from the factory. *Testing takes place on a per-chip basis*, so a chip has to pass tight testing before it gets sold to the customer. That is important because even the smallest number of faulty chips can result in enormous costs and reputational damage for companies. <br />

Debugging after fabrication is another potential time burden. Every minute or even hour spent identifying and rectifying faults could incur astronomical opportunity costs as schedules may be blown for productions, product releases postponed, and shipments of faulty components might bring down an entire business due to the resulting loss of customer confidence and eventual collapse into bankruptcy. <br />

Defects in chips are intrinsically physical problems and constitute the root cause of challenges addressed through Design for Testability (DFT). DFT is a methodology that integrates testing features into the design of a chip, making it easier to detect and diagnose defects. CMOS (Complementary Metal-Oxide-Semiconductor) fabrication, a widely used process in chip manufacturing, is highly complex and involves several stages. <br />

One of the most critical steps is photolithography, in which patterns are implanted onto silicon wafers. At this stage, even slight mistakes can create physical deviations and defects in the chip. The commonest ones include open defects, where there is no or incomplete and broken contact between two components. These defects often appear after post-silicon manufacturing processes or in the case of returned defective chips by customers. After the primary wafer fabrication process, dedicated testing teams, using high-end, expensive test equipment, find defects. This equipment includes high-resolution microscopes and cameras capable of scanning large areas of the chip with precision. The captured images are analyzed to pinpoint areas of concern and initiate further investigations. <br />

Once defective chips are identified, Physical Failure Analysis (PFA) is conducted to determine the root cause of the problem. PFA is a meticulous process that examines the defect’s origin, helping manufacturers enhance production quality and refine processes to prevent future issues. Common defects in chips include misaligned layers, contamination, regions that may be incompletely etched or over-etched; these defects significantly affect chip performance and reliability. Using DFT methods and more sophisticated testing technologies is imperative for improving the yield, reliability, and cost effectiveness of VLSI fabrication.

## Proposed Approach: CNN-Based LSI Layout Analysis
This project approach is based on Convolutional Neural Networks (CNNs) that analyze LSI layout images and determine failure-prone segments. This approach will start with the image preprocessing stage where the LSI layout images are segmented, normalized, and features extracted to achieve uniformity of the input for improving the performance of the model. Finally, an application-specific CNN architecture is proposed and defined that extracts spatial features with convolutional and pooling layers for segmentation of regions at various risk levels. It then uses labeled images, categorizing the different segments into being high or low-risk as classified based on failure histories. Other strategies applied during the process are data augmentation to reduce overfitting and generalization capabilities as well as regularization to generalize and minimize overfitting. After training, the CNN automatically extracts key features from input images and classifies segments, pointing out potential high-risk areas that can lead to failure. In addition to increasing the efficiency of layout analysis, it allows for the proactive identification and mitigation of defects and, therefore, improves yield, reliability, and production quality in semiconductor manufacturing.

### Model Architecture
The CNN used in the work is an effective architecture which is designed specifically to be implemented for defect detection and location in post-manufacturing wafers of VLSI. The size of the input images is grayscale images of 48x48 pixels in size and the defect coordinate is the output for a given wafer image. The detailed description about CNN model including its structure and functionality and the motivations of each design choice is given as below.<br />

The CNN model is built sequentially, consisting of:
 1. Input Layer to process images with dimensions 48×48×1.
 2. Convolutional Layers for extracting spatial features.
 3. Max Pooling Layers for downsampling feature maps and reducing computation.
 4. Fully Connected Layers to aggregate extracted features for classification.
 5. Output Layer with 2304 nodes, corresponding to each grid cell in the 48x48 image.

![Screenshot 2024-12-16 090339](https://github.com/user-attachments/assets/0b31b86f-543f-468f-86d1-ec16b43e2b88)
### Layer-Wise Explanation
 ***Input Layer:*** <br />
 The input to the model is a 48x48 grayscale image, reshaped into dimensions 48×48×1.
 Grayscale images were considered because they are relatively simple but have enough detail
 to identify defect locations, and processing is computationally less demanding than an RGB
 image.<br />
 
 ***Convolutional Layers:*** <br />
 These layers can detect spatial features such as edges, corners, and texture by convolution over
 the input image or feature maps using filters also known as kernels.
 There are three convolutional layers, Layer 1: 32 filters size 3×3. Layer 2: 64 filters size 3×3.
 Layer 3: 128 filters size 3×3.<br />
 Operation: Each filter overlays input and calculates dot products for generating feature maps.
 ReLU (f(x)=max(0,x)) is an activation function that introduces nonlinearity so that it may
 learn complex patterns.<br />
 Reasoning: Convolutional layers are used rather than dense layers for spatial data as they
 retain the spatial hierarchy of features but are not costly in terms of parameters.<br />
 
 ***Max Pooling Layers:*** <br />
 Objective: Pooling layers reduce spatial dimensions of feature maps so dominant features are
 retained, while simultaneously decreasing the computational load, preventing overfitting. Convolutional layers are followed by a pooling layer. There are two types of windows in the pooling
 layer 2x2 and stride=2.<br />
 Pooling window operation: every window selects the maximum feature of its region.<br />
 Rationale: Max pooling outperforms alternatives like average pooling in capturing critical
 features, especially in defect detection where prominent features are crucial.
 
 ***Flatten Layer:*** <br />
 After the final pooling layer, the multidimensional feature maps (shape 4×4×128) are flattened
 into a one-dimensional vector of size 2048. This transformation is necessary to connect the
 convolutional layers to fully connected layers.<br />
 
 ***Fully Connected Layers:*** <br />
 Purpose: These layers aggregate the learned features to make final predictions. The model
 employs a single fully connected layer with 256 neurons and ReLU activation. There is a
 dropout rate of 50% applied to the model to prevent overfitting by randomly disabling some
 neurons during training.<br />
 Rationale: Fully connected layers combine all the learned features so that it enables the network
 to learn complex combinations of spatial patterns relevant to defect localization.<br />
 
 ***Output Layer:*** <br />
 The output layer consists of 2304 neurons, which represent each of the grid cells in the 48x48
 image. The softmax activation function transforms the raw scores into probabilities, meaning
 the probability of the defect existing in each of the grid cells.<br />
 Rationale: Softmax ensures that the sum of probabilities equals 1, hence it is suitable for
 multi-class classification tasks.
 
## Results
Model Performance : <br />

![9 para](https://github.com/user-attachments/assets/aec2f3af-93f7-4d35-ae3f-c24437a7aa7a)
![test result img ](https://github.com/user-attachments/assets/19808672-bbb5-4a07-99ad-a0031964d99a)
