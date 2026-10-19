import cv2 as cv
import numpy as np
import glob
import os

# ============================================================
# PRACTICAL 1 - CAMERA CALIBRATION USING OPENCV
# ============================================================

# Objective:
# Estimate intrinsic camera parameters using OpenCV.

# Checkerboard:
# example 9 x 6 INNER CORNERcccS, Normally this means 10 x 7 squares.
# example 8 x 5 INNER CORNERS, Normally this means 9 x 6 squares.

# Controls:
# C or c = Capture image
# Q or q = Quit / Finish

# ============================================================

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

CHECKERBOARD = (8, 5)

CALIBRATION_FOLDER = "calibration_images"

# Create folder if it does not already exist
os.makedirs(CALIBRATION_FOLDER, exist_ok=True)

# ============================================================
# FUNCTION 1 - OPEN A WORKING WEBCAM
# ============================================================

def open_camera():

    print("\nSearching for webcam...")

    # Try camera indexes 0 to 3
    for camera_index in range(4):

        print(f"Trying camera {camera_index}...")

        # DirectShow is generally reliable on Windows
        cap = cv.VideoCapture(
            camera_index,
            cv.CAP_DSHOW
        )

        if cap.isOpened():

            # Test whether an actual frame can be obtained
            ret, frame = cap.read()

            if ret and frame is not None:

                print(
                    f"Camera {camera_index} opened successfully."
                )

                return cap

        cap.release()

    print("\nERROR: No working webcam was found.")

    return None


# ============================================================
# STEP 1 & STEP 2
# CAPTURE 15-20 CHECKERBOARD IMAGES
# ============================================================

def capture_images():

    cap = open_camera()

    if cap is None:
        return

    image_count = 0

    print("\n========================================")
    print(" CAMERA CALIBRATION - IMAGE CAPTURE")
    print("========================================")

    print("\nCheckerboard:")
    print("9 x 6 INNER CORNERS")

    print("\nControls:")
    print("C or c = Capture image")
    print("Q or q = Finish")

    print("\nCapture approximately 15-20 images.")

    print("\nUse different:")
    print("- angles")
    print("- distances")
    print("- positions")
    print("- orientations")

    print("\nIMPORTANT:")
    print("Keep the complete checkerboard visible.")
    print("Click the webcam window before pressing C or Q.")

    print("========================================\n")

    while True:

        # ----------------------------------------------------
        # READ FRAME FROM WEBCAM
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret or frame is None:

            print(
                "ERROR: Could not read webcam frame."
            )

            break

        # Copy frame for drawing
        display = frame.copy()

        # ----------------------------------------------------
        # CONVERT TO GRAYSCALE
        # ----------------------------------------------------

        gray = cv.cvtColor(
            frame,
            cv.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # STEP 3
        # DETECT CHECKERBOARD CORNERS
        # ----------------------------------------------------

        found, corners = cv.findChessboardCornersSB(
            gray,
            CHECKERBOARD
        )

        # ----------------------------------------------------
        # CHECKERBOARD DETECTED
        # ----------------------------------------------------

        if found:

            cv.drawChessboardCorners(
                display,
                CHECKERBOARD,
                corners,
                found
            )

            cv.putText(
                display,
                "CHECKERBOARD DETECTED",
                (20, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv.putText(
                display,
                "Press C to capture",
                (20, 75),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # ----------------------------------------------------
        # CHECKERBOARD NOT DETECTED
        # ----------------------------------------------------

        else:

            cv.putText(
                display,
                "CHECKERBOARD NOT DETECTED",
                (20, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv.putText(
                display,
                "Adjust board position/distance",
                (20, 75),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # DISPLAY IMAGE COUNT
        # ----------------------------------------------------

        cv.putText(
            display,
            f"Images captured: {image_count}",
            (20, 115),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # DISPLAY CONTROLS
        # ----------------------------------------------------

        cv.putText(
            display,
            "C = Capture Q = Finish",
            (20, 155),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # ----------------------------------------------------
        # SHOW WEBCAM WINDOW
        # ----------------------------------------------------

        cv.imshow(
            "Camera Calibration - Image Capture",
            display
        )

        # Read keyboard
        key = cv.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # C/c = CAPTURE IMAGE
        # ----------------------------------------------------

        if key in (ord('c'), ord('C')):

            if found:

                filename = os.path.join(
                    CALIBRATION_FOLDER,
                    f"image_{image_count:02d}.jpg"
                )

                success = cv.imwrite(
                    filename,
                    frame
                )

                if success:

                    image_count += 1

                    print(
                        f"Image {image_count} "
                        f"saved successfully."
                    )

                else:

                    print(
                        "ERROR: Image could not be saved."
                    )

            else:

                print(
                    "Image NOT captured: "
                    "checkerboard was not detected."
                )

        # ----------------------------------------------------
        # Q/q = FINISH
        # ----------------------------------------------------

        elif key in (ord('q'), ord('Q')):

            print("\nImage capture finished.")

            break

    # --------------------------------------------------------
    # RELEASE CAMERA
    # --------------------------------------------------------

    cap.release()

    cv.destroyAllWindows()

    print(
        f"\nTotal images captured: {image_count}"
    )


# ============================================================
# STEP 3 & STEP 4
# DETECT CORNERS AND CALIBRATE CAMERA
# ============================================================

def calibrate_camera():

    print("\n========================================")
    print(" CAMERA CALIBRATION")
    print("========================================")

    # --------------------------------------------------------
    # CREATE 3D OBJECT POINTS
    # --------------------------------------------------------

    # The checkerboard is flat.

    # Therefore Z = 0.

    # Points:

    # (0,0,0), (1,0,0), (2,0,0), ...
    # (0,1,0), (1,1,0), (2,1,0), ...

    # --------------------------------------------------------

    objp = np.zeros(
        (
            CHECKERBOARD[0] * CHECKERBOARD[1],
            3
        ),
        dtype=np.float32
    )

    objp[:, :2] = np.mgrid[
        0:CHECKERBOARD[0],
        0:CHECKERBOARD[1]
    ].T.reshape(-1, 2)

    # --------------------------------------------------------
    # ARRAYS FOR CALIBRATION POINTS
    # --------------------------------------------------------

    # 3D real-world points
    objpoints = []

    # 2D image points
    imgpoints = []

    # --------------------------------------------------------
    # LOAD CAPTURED IMAGES
    # --------------------------------------------------------

    images = sorted(
        glob.glob(
            os.path.join(
                CALIBRATION_FOLDER,
                "*.jpg"
            )
        )
    )

    if len(images) == 0:

        print(
            "\nERROR: No calibration images found."
        )

        return None

    print(
        f"\nNumber of images found: {len(images)}"
    )

    print(
        "\nDetecting checkerboard corners..."
    )

    gray = None

    valid_images = 0

    # --------------------------------------------------------
    # PROCESS EACH IMAGE
    # --------------------------------------------------------

    for filename in images:

        img = cv.imread(filename)

        if img is None:

            print(
                filename,
                "-- Could not read image"
            )

            continue

        # Convert image to grayscale
        gray = cv.cvtColor(
            img,
            cv.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # DETECT CHECKERBOARD
        # ----------------------------------------------------

        found, corners = cv.findChessboardCornersSB(
            gray,
            CHECKERBOARD
        )

        if found:

            # Store 3D object points
            objpoints.append(
                objp.copy()
            )

            # ------------------------------------------------
            # Ensure corner points are float32
            # ------------------------------------------------

            corners = np.asarray(
                corners,
                dtype=np.float32
            )

            # Store 2D image points
            imgpoints.append(
                corners
            )

            valid_images += 1

            # Draw checkerboard corners
            cv.drawChessboardCorners(
                img,
                CHECKERBOARD,
                corners,
                found
            )

            cv.imshow(
                "Detected Checkerboard Corners",
                img
            )

            cv.waitKey(300)

            print(
                filename,
                "-- Checkerboard detected"
            )

        else:

            print(
                filename,
                "-- Checkerboard NOT detected"
            )

    cv.destroyAllWindows()

    # --------------------------------------------------------
    # CHECK NUMBER OF VALID IMAGES
    # --------------------------------------------------------

    print(
        f"\nValid calibration images: {valid_images}"
    )

    if valid_images < 5:

        print(
            "\nERROR: Not enough valid images."
        )

        print(
            "Capture at least 10-15 good "
            "checkerboard images."
        )

        return None

    # ========================================================
    # STEP 4
    # CAMERA CALIBRATION
    # ========================================================

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n========================================")
    print(" CAMERA CALIBRATION RESULTS")
    print("========================================")

    # --------------------------------------------------------
    # RMS CALIBRATION ERROR
    # --------------------------------------------------------

    print("\nRMS Calibration Error:")

    print(ret)

    # --------------------------------------------------------
    # CAMERA MATRIX
    # --------------------------------------------------------

    print("\nCamera Matrix:")

    print(mtx)

    print("\nCamera Matrix format:")

    print("""
    | fx 0 cx |
    K = | 0 fy cy |
    | 0 0 1 |
    """)

    print(
        "fx =",
        mtx[0, 0]
    )

    print(
        "fy =",
        mtx[1, 1]
    )

    print(
        "cx =",
        mtx[0, 2]
    )

    print(
        "cy =",
        mtx[1, 2]
    )

    # --------------------------------------------------------
    # DISTORTION COEFFICIENTS
    # --------------------------------------------------------

    print(
        "\nDistortion Coefficients:"
    )

    print(dist)

    # --------------------------------------------------------
    # ROTATION VECTORS
    # --------------------------------------------------------

    print(
        "\nRotation Vectors:"
    )

    for i, rvec in enumerate(rvecs):

        print(
            f"Image {i + 1}:",
            rvec.flatten()
        )

    # --------------------------------------------------------
    # TRANSLATION VECTORS
    # --------------------------------------------------------

    print(
        "\nTranslation Vectors:"
    )

    for i, tvec in enumerate(tvecs):

        print(
            f"Image {i + 1}:",
            tvec.flatten()
        )

    # ========================================================
    # MEAN REPROJECTION ERROR
    #
    # OpenCV 5 FIX:
    #
    # findChessboardCornersSB may produce points in shape:
    #
    # (N, 2)
    #
    # while projectPoints may produce:
    #
    # (N, 1, 2)
    #
    # cv.norm() requires matching types/layouts.
    #
    # Therefore BOTH arrays are converted to:
    #
    # dtype = float32
    # shape = (N, 2)
    #
    # ========================================================

    total_error = 0.0

    for i in range(len(objpoints)):

        # ----------------------------------------------------
        # PROJECT 3D POINTS BACK INTO IMAGE
        # ----------------------------------------------------

        projected_points, _ = cv.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            mtx,
            dist
        )

        # ----------------------------------------------------
        # FIX TYPE AND SHAPE OF DETECTED POINTS
        # ----------------------------------------------------

        detected = np.asarray(
            imgpoints[i],
            dtype=np.float32
        ).reshape(-1, 2)

        # ----------------------------------------------------
        # FIX TYPE AND SHAPE OF PROJECTED POINTS
        # ----------------------------------------------------

        projected = np.asarray(
            projected_points,
            dtype=np.float32
        ).reshape(-1, 2)

        # ----------------------------------------------------
        # CALCULATE REPROJECTION ERROR
        # ----------------------------------------------------

        error = cv.norm(
            detected,
            projected,
            cv.NORM_L2
        ) / len(projected)

        total_error += error

    # Calculate mean error
    mean_error = (
        total_error / len(objpoints)
    )

    print(
        "\nMean Reprojection Error:"
    )

    print(mean_error)

    # ========================================================
    # SAVE CALIBRATION PARAMETERS
    # ========================================================

    np.savez(
        "camera_calibration.npz",
        camera_matrix=mtx,
        distortion_coefficients=dist,
        rotation_vectors=np.array(rvecs),
        translation_vectors=np.array(tvecs)
    )

    print(
        "\nCalibration parameters saved as:"
    )

    print(
        "camera_calibration.npz"
    )

    return mtx, dist


# ============================================================
# STEP 5
# REMOVE LENS DISTORTION
# ============================================================

def undistort_camera(mtx, dist):

    cap = open_camera()

    if cap is None:
        return

    print("\n========================================")
    print(" UNDISTORTION DEMONSTRATION")
    print("========================================")

    print("\nLeft = Original image")
    print("Right = Undistorted image")

    print(
        "\nPress Q or q to exit."
    )

    while True:

        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret or frame is None:

            print(
                "ERROR: Could not read webcam frame."
            )

            break

        # ----------------------------------------------------
        # REMOVE DISTORTION
        # ----------------------------------------------------

        dst = cv.undistort(
            frame,
            mtx,
            dist,
            None,
            mtx
        )

        # ----------------------------------------------------
        # LABEL ORIGINAL IMAGE
        # ----------------------------------------------------

        cv.putText(
            frame,
            "Original",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # LABEL UNDISTORTED IMAGE
        # ----------------------------------------------------

        cv.putText(
            dst,
            "Undistorted",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # DISPLAY SIDE BY SIDE
        # ----------------------------------------------------

        comparison = np.hstack(
            (
                frame,
                dst
            )
        )

        cv.imshow(
            "Original vs Undistorted",
            comparison
        )

        # Read keyboard
        key = cv.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # Q/q = EXIT
        # ----------------------------------------------------

        if key in (
            ord('q'),
            ord('Q')
        ):

            print(
                "\nUndistortion demonstration finished."
            )

            break

        # --------------------------------------------------------
        # RELEASE CAMERA
        # --------------------------------------------------------

    cap.release()

    cv.destroyAllWindows()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    while True:

        print("""
================================================
PRACTICAL 1 - CAMERA CALIBRATION
================================================

Objective:
Estimate intrinsic camera parameters using OpenCV.

Checkerboard:
9 x 6 INNER CORNERS
(normally 10 x 7 squares)

------------------------------------------------

1 - Capture checkerboard images

2 - Calibrate using captured images

3 - Exit

------------------------------------------------

WEBCAM CONTROLS

C or c = Capture image

Q or q = Finish / Quit

================================================
        """)

        choice = input(
            "Enter your choice (1/2/3): "
        )

        # ====================================================
        # OPTION 1
        # CAPTURE IMAGES
        # ====================================================

        if choice == "1":

            capture_images()

            answer = input(
                "\nDo you want to calibrate "
                "the camera now? (Y/N): "
            )

            if answer.lower() == "y":

                result = calibrate_camera()

                if result is not None:

                    mtx, dist = result

                    answer2 = input(
                        "\nShow undistortion "
                        "demonstration? (Y/N): "
                    )

                    if answer2.lower() == "y":

                        undistort_camera(
                            mtx,
                            dist
                        )

        # ====================================================
        # OPTION 2
        # CALIBRATE EXISTING IMAGES
        # ====================================================

        elif choice == "2":

            result = calibrate_camera()

            if result is not None:

                mtx, dist = result

                answer = input(
                    "\nShow undistortion "
                    "demonstration? (Y/N): "
                )

                if answer.lower() == "y":

                    undistort_camera(
                        mtx,
                        dist
                    )

        # ====================================================
        # OPTION 3
        # EXIT
        # ====================================================

        elif choice == "3":

            print(
                "\nProgram terminated."
            )

            break

        # ====================================================
        # INVALID INPUT
        # ====================================================

        else:

            print(
                "\nInvalid choice. "
                "Please enter 1, 2 or 3."
            )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()

