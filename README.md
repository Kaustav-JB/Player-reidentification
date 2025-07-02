# Player Re-Identification in Sports Video using YOLOv11 and Deep SORT

This project demonstrates a real-time player tracking and re-identification system for sports analytics. The objective was to consistently assign and retain unique player IDs in a short video clip, even if players temporarily leave the frame.

## 🎯 Objective

- Detect players and goalkeepers in a football video using a pre-trained object detection model.
- Track each player across frames with consistent IDs.
- Handle occlusions and re-identify players upon return.

## 🛠️ Technologies Used

- **YOLOv11** (custom fine-tuned model, not included in this repo)
- **Deep SORT** (for re-identification and tracking)
- **OpenCV** (for video I/O and visualization)
- **NumPy**

## 🗂️ Project Structure
Player-reidentification/ </br>
├── main.py  _#Detection + tracking pipeline_ </br>
├── output_tracked.mp4  _#Output with player IDs visualized_ </br>
├── requirements.txt  _#Dependency list_ </br>
└── README.md  _#Project overview_ </br>

⚠️ **Note**: The model used for detection is proprietary and not included in this repository.

## 📦 Requirements
`` pip install -r requirements.txt ``

## 📌 Notes

 - The tracker is configured to recognize only relevant player classes.
 - Bounding box size, shape, and confidence thresholds help reduce false detections.
 - The Deep SORT algorithm is tuned for identity retention with moderate occlusion handling.
