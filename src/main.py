from config import *
from detect_cars import CarDetector
from image_processing import ImageProcessor
from image_utils import ImageUtils
from utils import get_logger
import os

logger = get_logger("Main")

def get_parking_spaces_status_from_images(image_path: str) -> list[bool]:
    image = ImageUtils.load_image(image_path)
    image = ImageUtils.resize_with_aspect_ratio(image, width=1000)

    car_rectangles = CarDetector().detect(image)
    car_centers = [rect.get_center() for rect in car_rectangles]
    return ImageProcessor.check_parking_spaces(image, car_centers, display_intermediate=DISPLAY_PARKING_SPACES_INTERMEDIATE)

def query_parking_spaces_from_image(image_path: str, query_path: str, results_path:str) -> None:
    """ 
    Queries the parking spaces status from images and writes results to a file. 
    Query file contains on first row how many queries there are.
    Each query is a number representing the parking space index (starting from 1).
    Results file will contain the number of queries and for each query it's index and status (1 for occupied, 0 for free).
    """
    parking_spaces = get_parking_spaces_status_from_images(image_path)
    
    with open(query_path, 'r') as file:
        queries = file.readlines()
    queries = [query.strip() for query in queries if query.strip()]

    for i in range(1, len(queries)):
        status = parking_spaces[int(queries[i]) - 1] if queries[i].isdigit() else False
        status = 1 if status else 0
        queries[i] += f" {status}"

    with open(results_path, 'w') as file:
        for query_row in queries:
            file.write(f"{query_row}\n")
            
def get_task1_results(dataset_folder: str, save_to_folder:str, remove_old_results=False) -> None:
    """
    Processes all images in the dataset folder, queries parking spaces, and saves results to a specified folder.
    """
    if not os.path.exists(save_to_folder):
        os.makedirs(save_to_folder)
        
    if remove_old_results:
        for file in os.listdir(save_to_folder):
            if file.endswith("_results.txt"):
                os.remove(os.path.join(save_to_folder, file))

    for image_name in os.listdir(dataset_folder):
        if image_name.endswith(".jpg"):
            image_name = image_name.split(".")[0]
            image_path = os.path.join(dataset_folder, image_name + ".jpg")
            query_path = os.path.join(dataset_folder, f"{image_name}_query.txt")
            results_path = os.path.join(save_to_folder, f"{image_name}_results.txt")
            query_parking_spaces_from_image(image_path, query_path, results_path)

def main() -> int:
    # dataset_folder = "train/Task1"
    # save_to_folder = "train/Task1/results"
    # get_task1_results(dataset_folder, save_to_folder, remove_old_results=True)
        
    parking_spaces = get_parking_spaces_status_from_images(SELECTED_IMAGE)
    for i in range(len(parking_spaces) -1, -1, -1):
        logger.info(f"Parking space {i + 1}: {'Occupied' if parking_spaces[i] else 'Free'}")
    image = ImageUtils.load_image(SELECTED_IMAGE)
    image = ImageUtils.resize_with_aspect_ratio(image, width=1000)
    car_rectangles = CarDetector().detect(image)
    # CarDetector().draw_car_boxes(image, car_rectangles)
    CarDetector().draw_car_centers(image, car_rectangles)
    ImageUtils.draw_box_on_image(image, ImageProcessor.PARKING_BOX)
    ImageUtils.display(image, title="Parking Spaces", display=True, size=1000)

if __name__ == "__main__":
    exit(main())
