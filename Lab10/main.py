import cv2
import time
from collections import deque

# ---------------------------------------------------
# Function to create tracker
# ---------------------------------------------------
def create_tracker():

    # Newer OpenCV versions
    if hasattr(cv2, "TrackerCSRT_create"):
        print("Using CSRT tracker")
        return cv2.TrackerCSRT_create()

    # Some OpenCV versions expose trackers through legacy
    if hasattr(cv2, "legacy"):
        if hasattr(cv2.legacy, "TrackerCSRT_create"):
            print("Using Legacy CSRT tracker")
            return cv2.legacy.TrackerCSRT_create()

    print("CSRT tracker is not available.")
    print("Install OpenCV Contrib using:")
    print("pip install opencv-contrib-python")

    return None


# ---------------------------------------------------
# Initialise webcam
# ---------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam could not be opened.")
    exit()

tracker = None
tracking = False

# Store previous centre points to draw trajectory
trajectory = deque(maxlen=50)

print("R : Select an object")
print("Q : Quit")


# ---------------------------------------------------
# Main processing loop
# ---------------------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Error reading webcam frame.")
        break

    start_time = time.time()

    # ------------------------------------------------
    # Update tracker
    # ------------------------------------------------
    if tracking:

        success, bbox = tracker.update(frame)

        if success:

            x, y, w, h = [int(v) for v in bbox]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Calculate object centre
            cx = x + w // 2
            cy = y + h // 2

            trajectory.append((cx, cy))

            # Draw centre point
            cv2.circle(
                frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                "TRACKING",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "TRACKING LOST",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    # ------------------------------------------------
    # Draw trajectory
    # ------------------------------------------------
    for i in range(1, len(trajectory)):

        cv2.line(
            frame,
            trajectory[i - 1],
            trajectory[i],
            (255, 0, 0),
            2
        )

    # ------------------------------------------------
    # Calculate FPS
    # ------------------------------------------------
    elapsed = time.time() - start_time

    if elapsed > 0:
        fps = 1.0 / elapsed
    else:
        fps = 0

    cv2.putText(
        frame,
        "FPS: {:.1f}".format(fps),
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "R: Select Object | Q: Quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.imshow("Real-Time Object Tracking", frame)

    key = cv2.waitKey(1) & 0xFF

    # ------------------------------------------------
    # Select / re-select object
    # ------------------------------------------------
    if key == ord("r"):

        bbox = cv2.selectROI(
            "Select Object",
            frame,
            fromCenter=False,
            showCrosshair=True
        )

        cv2.destroyWindow("Select Object")

        if bbox[2] > 0 and bbox[3] > 0:

            tracker = create_tracker()
            tracker.init(frame, bbox)

            tracking = True
            trajectory.clear()

            print("Object selected. Tracking started.")

    # Quit
    elif key == ord("q"):
        break


# ---------------------------------------------------
# Release resources
# ---------------------------------------------------
cap.release()
cv2.destroyAllWindows()