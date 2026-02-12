import cv2
try:
    from deepface import DeepFace
except ImportError:
    print("\nCRITICAL ERROR: 'deepface' library not found.")
    print("Please wait for the installation to finish or run: pip install -r requirements.txt\n")
    exit(1)
import logging
import cv2.data

logging.basicConfig(level=logging.INFO)

def main():

  
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logging.error("Could not open webcam. Please ensure a camera is connected.")
        return

    logging.info("Starting emotion recognition. Press 'q' to quit.")
    logging.info("Note: The first time you run this, DeepFace will download the necessary model weights. This may take a few minutes.")

 
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            logging.error("Failed to capture frame from webcam.")
            break

        frame = cv2.flip(frame, 1)

        try:
           
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            
            if isinstance(results, dict):
                results = [results]
            
            for face in results:
                region = face.get('region', {})
                x = region.get('x', 0)
                y = region.get('y', 0)
                w = region.get('w', 0)
                h = region.get('h', 0)
                
             
                emotion = face.get('dominant_emotion')
                
               
                if w > 0 and h > 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    if emotion:
                        cv2.putText(frame, str(emotion).upper(), (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        except Exception as e:
            logging.error(f"Error during analysis: {e}")

        cv2.imshow('Emotion Recognition AI', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

