import cv2
try:
    from deepface import DeepFace
except ImportError:
    print("\nCRITICAL ERROR: 'deepface' library not found.")
    print("Please wait for the installation to finish or run: pip install -r requirements.txt\n")
    exit(1)
import logging
import cv2.data

# Configure logging to show information about the process
logging.basicConfig(level=logging.INFO)

def main():
    """
    Main function to run the emotion recognition AI.
    """
    # Initialize the webcam
    # 0 is usually the default camera. If you have multiple, try 1, 2, etc.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logging.error("Could not open webcam. Please ensure a camera is connected.")
        return

    logging.info("Starting emotion recognition. Press 'q' to quit.")
    logging.info("Note: The first time you run this, DeepFace will download the necessary model weights. This may take a few minutes.")

    # Face detection classifier from OpenCV (Haar Cascades)
    # This is used for faster face detection to draw rectangles, 
    # although DeepFace also detects faces, using a separate detector here 
    # allows us to only run DeepFace when a face is present or to optimize.
    # However, DeepFace.analyze already handles detection. 
    # Letting DeepFace handle everything for simplicity and accuracy.
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            logging.error("Failed to capture frame from webcam.")
            break

        # Flip the frame horizontally for a specialized mirror effect
        frame = cv2.flip(frame, 1)

        try:
            # Analyze the frame for emotion
            # actions=['emotion'] specifies we only want emotion analysis
            # enforce_detection=False prevents the code from crashing if no face is detected
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            
            # Ensure results is a list (DeepFace can return a dict for single face)
            if isinstance(results, dict):
                results = [results]
            
            for face in results:
                # The 'region' key contains the face coordinates
                region = face.get('region', {})
                x = region.get('x', 0)
                y = region.get('y', 0)
                w = region.get('w', 0)
                h = region.get('h', 0)
                
                # If DeepFace says confidence is low (it doesn't explicitly return confidence in the top level dict usually, 
                # but results implies detection was successful if enforce_detection=False returned something valid not empty).
                # Actually if no face found with enforce_detection=False, it might return a full frame analysis or dummy.
                # But usually it tries its best.
                
                # Get the dominant emotion
                emotion = face.get('dominant_emotion')
                
                # Draw the rectangle around the face
                # Only draw if dimensions seem valid (not full screen if detecting 'no face')
                if w > 0 and h > 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Display the emotion label
                    if emotion:
                        cv2.putText(frame, str(emotion).upper(), (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        except Exception as e:
            # Log error but don't crash the loop
            logging.error(f"Error during analysis: {e}")

        # Display the resulting frame
        cv2.imshow('Emotion Recognition AI', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
