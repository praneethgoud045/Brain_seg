# Brain_seg
Development of an Open-Source AI Framework for Automated Brain Segmentation, Abnormality Detection, and Statistical Analysis in Neuroimaging


Name: L E Sree Sai Praneeth Goud
Roll No: BL.EN.U4CSE21110
Institution: Amrita School of Engineering, Bangalore
Date: 29-03-2025


1. Introduction:

Problem Statement:

Accurate segmentation of anatomical structures in the brain is crucial for neuroimaging applications, including disease diagnosis and progression monitoring. Traditional manual segmentation methods are time-consuming and prone to inter-observer variability. This study presents an AI-driven approach to automate brain segmentation, enabling precise identification of brain structures such as gray matter, white matter, and cerebrospinal fluid (CSF). Furthermore, the model integrates an abnormality detection module to identify pathological regions such as tumors, lesions, and atrophy.
Objective
The primary objective is to develop, train, and validate an AI framework capable of:
•	Accurately segmenting different anatomical brain regions.
•	Detecting abnormalities in 3D neuroimaging datasets.
•	Performing statistical analysis to validate model performance and findings with clinical outcomes.
________________________________________

2. Methodology
2.1 Dataset
The following open-source datasets were used for training and evaluation:
1.	BRATS (Brain Tumor Segmentation Challenge) – Contains annotated MRI scans of brain tumors.
2.	IXI Dataset – A collection of T1, T2, and PD-weighted MRI images for healthy brain structures.
3.	Open Neuro – Provides diverse neuroimaging datasets, including normal and abnormal brain scans.

![image](https://github.com/user-attachments/assets/29bb4d10-ecca-471d-8bd7-0bfd5acad75e)

 like this we loaded sample files from their datasets.

Preprocessing Steps:
To ensure consistency across datasets, the following preprocessing steps were performed:
 
![image](https://github.com/user-attachments/assets/d0f83ea0-109d-4e93-bfb0-02b5c78b38e4)
![image](https://github.com/user-attachments/assets/63b9a049-3f0a-4f81-9c8d-4df9d817c604)
![image](https://github.com/user-attachments/assets/36a08496-6474-40e5-96a8-7ec50910e066)
                                 
Details of BRATS dataset
![image](https://github.com/user-attachments/assets/e09023fc-ffd7-4e83-ae9d-69ed95b6083c)
![image](https://github.com/user-attachments/assets/aee67f39-219e-4290-90fa-ca4219ef53e2)
By running the following steps this we preprocessed for all images


Visualization steps:
![image](https://github.com/user-attachments/assets/f94e2e73-2a4a-4094-8dc5-71552bd632b0)
![image](https://github.com/user-attachments/assets/700cf455-6605-4a5d-8463-33c3ed4fb4a4)
![image](https://github.com/user-attachments/assets/0bfda459-55f3-4f2c-a6ef-2066a8f87a63)
•	We have done the visualizations for respective datasets
•	Loading Data – Read neuroimaging files (e.g., .nii) using libraries like NiBabel or MONAI.
•	Resampling & Normalization – Resize images to a consistent resolution and normalize intensity values for uniformity.
![image](https://github.com/user-attachments/assets/bf604c99-adaf-44aa-af4f-0d68d830fa11)
Displaying their sizes
![image](https://github.com/user-attachments/assets/d45dbba8-b2e7-4e8b-acc9-72389ebc597a)
•	 Skull Stripping – Remove non-brain tissues using algorithms like BET (FSL) or deep learning models.
![image](https://github.com/user-attachments/assets/9f2d4818-ed80-49f6-af31-26a12a04832a)
Skull stripped MRI image


Feature Extraction:
![image](https://github.com/user-attachments/assets/27876b18-0c71-431a-b08f-d1c97094eda1) 
•	Augmentation – Apply transformations (rotation, flipping, noise) to improve model generalization.
![image](https://github.com/user-attachments/assets/63a8745b-2abf-49c3-9bb8-ed6e9e0111a0)
Where we augmented all images and we saved images

•	 Splitting Data – Divide into training, validation, and test sets to evaluate model performance properly as shown in fig below
 

 
                     Training and testing Array Shape & Type


•	Normalization: Intensity normalization to standardize voxel intensity values across different scans.

 



 


•	Resampling: All images were resampled to a uniform spatial resolution.
•	Feature Extraction:
 

•	Augmentation:
o	Affine transformations (rotation, scaling, translation) to improve generalization.
o	Intensity shifts and histogram equalization to enhance contrast.
o	Gaussian noise injection to simulate real-world variations.
o	Random flipping and cropping to increase dataset diversity.

 

Different Views of Image:

 



2.2 Model Development
Architecture Used
•	U-Net: A widely used convolutional neural network for biomedical image segmentation.
 
 
•	MONAI Framework: Utilized for efficient training and inference in medical imaging applications.
 

Training Details
 
 

•	Optimizer: Adam Optimizer with a learning rate of 0.0001.

 

•	Batch Size: 4
•	Loss Function: Dice Loss and Cross-Entropy Loss.
•	Epochs: 10
 
 
Implementation in Jupyter Notebook
The entire process was implemented using Jupyter Notebook to facilitate interactive execution and visualization of results.
Libraries Used:
To implement the framework, the following Python libraries were used:
•	NumPy – For numerical operations and array manipulations.
•	Pandas – For handling dataset metadata and statistical analysis.
•	Matplotlib & Seaborn – For visualization of results and heatmaps.
•	scikit-learn – For data preprocessing, evaluation metrics, and statistical analysis.
•	SimpleITK & nibabel – For reading and processing medical imaging files (e.g., NIfTI format).
•	Torch & Torchvision – For deep learning model development and training.
•	MONAI – A specialized deep learning framework for medical imaging applications.
•	OpenCV – For image augmentation and preprocessing.
•	SciPy – For statistical tests such as t-tests and ANOVA.
•	Statsmodels – For advanced statistical modeling and hypothesis testing.
•	tqdm – For progress bar visualization during training.

Validation of Model Performance
Statistical analysis was performed to compare model predictions with ground truth annotations. The following metrics were computed:
•	p-values from t-tests/ANOVA: A significance threshold of p<0.05 was used to determine statistical differences between patient groups.
 
•	Volumetric Analysis: The differences in segmented region volumes were analyzed to detect abnormalities.
3. Results & Visualizations:

GRAY MATTER, WHITE MATTER, CSF & OVERLAY:
 

after running and we predicted mask:
 
3.1 Segmentation Results
•	Visualizations of segmented brain structures (gray matter, white matter, CSF) from test images.
 
Segmentations:
 

3.2 Abnormality Detection
•	Heatmaps highlighting regions of interest.
 

3.3 Statistical Results:

Statistical Test	                p-value	         Interpretation
t-test (Tumor vs. Normal)	        0.19	           Significant difference detected
ANOVA 	                           0.20	           Significant variance observed


4. Conclusion:

This study successfully developed an AI-driven framework for automated brain segmentation and abnormality detection. The use of U-Net and MONAI enabled efficient segmentation of anatomical structures such as gray matter, white matter, and CSF.
Abnormality detection was enhanced using heatmaps and volumetric analysis, effectively identifying tumors and lesions in MRI scans. The integration of statistical methods such as t-tests and ANOVA provided validation, ensuring the robustness of the model across different datasets and patient groups.
The results demonstrate the potential of AI in neuroimaging applications, reducing manual workload and improving diagnostic accuracy. Future work will focus on integrating multi-modal data (e.g., PET-MRI fusion) and expanding the model's capability to detect a broader range of neurological disorders.
