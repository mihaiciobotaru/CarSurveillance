from ultralytics import YOLO
from config import *
from utils import Rectangle, get_logger
from image_utils import ImageUtils
import cv2
import numpy as np

logger = get_logger("CarDetector")

class CarDetector:
    """ ======= Class to detect cars in images using YOLOv8 model ======= """
    VEHICLE_CLASSES = {
        "car": 2,
        "truck": 7,
        "bus": 5,
        "motorcycle": 3
    }
    model = None
    verbose = False

    def __init__(self, model="NANO"):
        """ ======= Initialize the CarDetector with YOLOv8 medium model ======= """
        if model == "NANO":
            model_name = "yolov8n.pt"
        elif model == "MEDIUM":
            model_name = "yolov8m.pt"
        elif model == "LARGE":
            model_name = "yolov8l.pt"
        elif model == "X-LARGE":
            model_name = "yolov8x.pt"
        else:
            logger.warning(f"Invalid model type: {model}. Defaulting to NANO model.")
            model_name = "yolov8n.pt"
            
        logger.debug(f"Loading YOLOv8 model: {model_name}")

        self.model = YOLO(model_name)
        if LOGGING_LEVEL in ["DEBUG", "TRACE"]:
            self.verbose = True

    def detect(self, image:np.ndarray) -> list[Rectangle]:
        """ ======= Detect cars in the given image using YOLOv8 model ======= """
        logger.trace("Running car detection on the image.")

        try:
            results = self.model(image, verbose=self.verbose, conf=0.10, classes=list(self.VEHICLE_CLASSES.values()))
            logger.debug(f"Detected {len(results[0].boxes)} results.")
            if not results:
                logger.error("No results found.")
                return []

            car_rectangles = []
            for result in results:
                for box in result.boxes:
                    box_tuple = box.xyxy.cpu().numpy()[0]
                    for i in range(4):
                        box_tuple[i] = float(box_tuple[i])
                    rectangle = Rectangle.from_tuple(box_tuple)
                    car_rectangles.append(rectangle)
            return car_rectangles
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []

    def draw_car_rectangles(self, image, car_rectangles: list[Rectangle]) -> None:
        """======= Draw bounding rectangles around detected cars on the image ======= """
        for rectangle in car_rectangles:
            ImageUtils.draw_rectangle_on_image(image, rectangle, text='Car')

    def draw_car_centers(self, image, car_rectangles: list[Rectangle]) -> None:
        """ ======= Draw points at the center of detected cars on the image ======= """
        for rectangle in car_rectangles:
            car_center = rectangle.get_center()
            ImageUtils.draw_point_on_image(image, car_center, text='Car')

def main() -> int:
    """ ======= Main function to demonstrate car detection model ======= """
    detector = CarDetector()

    image = cv2.imread(SELECTED_FILE)
    if image is None:
        logger.error(f"Error: Could not read image from {SELECTED_FILE}")
        return 1
    
    car_rectangles = detector.detect(image)
    if not car_rectangles:
        logger.info("No cars detected.")
    else:
        detector.draw_car_rectangles(image, car_rectangles)
        cv2.imshow("Detected Cars", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return 0

if __name__ == "__main__":
    exit(main())
