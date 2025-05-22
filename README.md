# Facial Recognition Mall System

A comprehensive facial recognition system designed for mall security and visitor tracking, built with Python and AWS Rekognition.

## Features

- Real-time face detection and recognition
- AWS Rekognition integration for cloud-based face recognition
- Local face collection management
- Video processing capabilities
- Webcam face capture
- Duplicate face detection and cleanup
- Performance tracking and reporting
- Modern GUI interface

## Prerequisites

- Python 3.8 or higher
- AWS Account with Rekognition access
- Webcam (for face capture feature)
- Sufficient disk space for face storage

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/facial-recognition-mall.git
cd facial-recognition-mall
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure AWS credentials:
   - Create an AWS credentials file at `~/.aws/credentials`
   - Add your AWS access key and secret key:
```ini
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```

## Usage

1. Start the application:
```bash
python face_detection.py
```

2. Main Features:
   - Process Video (V): Analyze video files for face detection
   - Capture Face (C): Capture faces from webcam
   - Register Faces (R): Register multiple faces from a folder
   - Show Collection (S): View and manage face collection
   - Cleanup Duplicates (D): Find and remove duplicate faces
   - Index Faces (I): Review and index unrecognized faces

## Project Structure

```
facial-recognition-mall/
├── face_detection.py      # Main application file
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── README.md             # Project documentation
├── unrecognized_faces/   # Directory for new faces
├── indexed_faces/        # Directory for registered faces
├── collection_backups/   # Collection backup files
└── collection_exports/   # Collection export files
```

## AWS Configuration

1. Create a Rekognition Collection:
   - Collection ID: 'my-face-collection'
   - Region: ap-south-1 (or your preferred region)

2. Required AWS Permissions:
   - rekognition:CreateCollection
   - rekognition:DeleteCollection
   - rekognition:IndexFaces
   - rekognition:DeleteFaces
   - rekognition:SearchFacesByImage
   - rekognition:ListFaces

## Collaboration Guidelines

1. Branching Strategy:
   - `main`: Production-ready code
   - `develop`: Development branch
   - Feature branches: `feature/feature-name`
   - Bug fix branches: `fix/bug-name`

2. Commit Guidelines:
   - Use clear, descriptive commit messages
   - Reference issue numbers in commits
   - Keep commits focused and atomic

3. Pull Request Process:
   - Create PR from feature/fix branch to develop
   - Include description of changes
   - Request review from team members
   - Ensure all tests pass
   - Update documentation if needed

## Security Notes

- Never commit AWS credentials
- Keep face data secure and private
- Follow local privacy laws and regulations
- Regular security audits recommended

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers. 