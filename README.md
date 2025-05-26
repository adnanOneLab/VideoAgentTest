# Facial Recognition Tool

A facial recognition system that uses AWS Rekognition for face detection and recognition.

## Prerequisites

- Python 3.8 or higher
- AWS Account with Rekognition access
- AWS credentials configured (see Setup section)

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Configure AWS credentials:
   - Create an AWS account if you don't have one
   - Create an IAM user with Rekognition access
   - Configure AWS credentials using one of these methods:
     - AWS CLI: `aws configure`
     - Environment variables: Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
     - Or create `~/.aws/credentials` file with:
     ```
     [default]
     aws_access_key_id = YOUR_ACCESS_KEY
     aws_secret_access_key = YOUR_SECRET_KEY
     ```

3. Create an AWS Rekognition collection:
   - The application will automatically create a collection named 'my-face-collection'
   - Or you can create it manually in the AWS Console

## Running the Application

1. Run the main application:
```bash
python src/core/face_detection.py
```

2. Using the GUI:
   - Press 'C' to capture faces from webcam
   - Press 'V' to process a video file
   - Press 'R' to register faces from a folder
   - Press 'S' to show/manage the face collection
   - Press 'Q' to quit

## Features

- Face detection and recognition using AWS Rekognition
- Real-time face quality assessment
- Face capture from webcam
- Video processing
- Face collection management
- AWS collection synchronization
- Performance metrics and reporting

## Directory Structure

- `src/core/` - Core application code
- `src/utils/` - Utility functions
- `indexed_faces/` - Stored face images
- `unrecognized_faces/` - Temporary storage for unrecognized faces
- `output/` - Processed video output
- `performance_plots/` - Generated performance reports 