from ultralytics import YOLO

# 1. Load the YOLO11 Nano model. 
# (It will automatically download the 5MB 'yolo11n.pt' file the very first time you run this)
model = YOLO("yolo11n.pt")

print("Starting the camera... Press 'q' in the video window to quit.")

# 2. Run the model on your live webcam.
# source=0 means "use the default USB/laptop camera"
# show=True tells the model to automatically draw boxes and show the video window
# stream=True processes the video frame-by-frame efficiently
results = model.predict(source=0, show=True, stream=True)

# 3. Start the continuous loop to keep reading frames
for frame_result in results:
    # Right now, 'show=True' is doing all the drawing for us.
    # But later, this is the exact spot where we will extract the math 
    # (where the box is) to tell your Raspberry Pi how to drive!
    pass
