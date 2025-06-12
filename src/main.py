from config import *
from detect_cars import CarDetector
from image_processor import ImageProcessor
from image_utils import ImageUtils
from video_utils import VideoUtils
from utils import get_logger
import numpy as np
import os

logger = get_logger("Main")

def get_parking_spaces_status_from_image(image_path: str|np.ndarray, display_intermediate=False) -> list[bool]:
    """ ======= Get the status of each parking space from a given image ======= """
    if isinstance(image_path, str):
        image = ImageUtils.load_image(image_path)
    else:
        image = image_path # Already an image

    image = ImageUtils.resize_with_aspect_ratio(image, width=1000)
    if display_intermediate:
        ImageUtils.display(image, title="Original Image", display=True, size=1000)

    car_rectangles = CarDetector().detect(image)
    car_centers = [rect.get_center() for rect in car_rectangles]
    if display_intermediate:
        CarDetector().draw_car_centers(image, car_rectangles)
        ImageUtils.draw_quadrilateral_on_image(image, ImageProcessor.PARKING_BOX)
        ImageUtils.display(image, title="Detected Cars", display=True, size=1000)
    return ImageProcessor.check_parking_spaces(image, car_centers, display_intermediate=DISPLAY_PARKING_SPACES_INTERMEDIATE)

def task4(video_path: str) -> None:
    """ 
    ======= Query parking spaces status from an image and write results to a file =======

    Queries the parking spaces status from images and writes results to a file. 
    Query file contains on first row how many queries there are.
    Each query is a number representing the parking space index (starting from 1).
    Results file will contain the number of queries and for each query it's index and status (1 for occupied, 0 for free).
    """
    last_frame = VideoUtils.get_last_frame(video_path)
    image = ImageUtils.load_image(last_frame)
    image = ImageUtils.resize_with_aspect_ratio(image, width=1000)

    detector = CarDetector("LARGE")
    cars = detector.detect(image)
    car_centers = [car.get_center() for car in cars]
    
    if not SAVE_TO_FOLDER_SWITCH:
        ImageUtils.draw_quadrilateral_on_image(image, ImageProcessor.TRAFFIC_LIGHT_QUEUE)
        for car in car_centers:
            ImageUtils.draw_point_on_image(image, car, text="Car")
        ImageUtils.display(image, title="Selected Image", display=True, size=1000)

    logger.info("Counting cars in the traffic light queue...")
    return ImageProcessor.count_cars_traffic_light_queue(image, car_centers, display_intermediate=DISPLAY_PARKING_SPACES_INTERMEDIATE)

def task2(video_path: str) -> None:
    """ 
    ======= Solve Project's Task 2 =======

    Processes a video file to detect parking spaces from the last frame
    """
    video_frames = VideoUtils.read_video_frames(video_path)
    last_frame = None
    for frame in video_frames:
        last_frame = frame

    if last_frame is not None:
        parking_statuses = get_parking_spaces_status_from_image(last_frame, False)
        logger.debug(f"Parking spaces status from video {video_path}: {parking_statuses}")
        return " ".join(['1' if status else '0' for status in parking_statuses])
    
    return []

def task3(image_path: str) -> None:
    pass

def task1(image_path: str, query_path: str) -> None:
    """
    ======= Solve Project's Task 1 =======

    Processes an image file to detect parking spaces and write results to a file.
    The query file contains the number of queries and each query is a number representing the parking space index (starting from 1).
    The results file will contain the number of queries and for each query its index and status (1 for occupied, 0 for free).
    """
    parking_spaces = get_parking_spaces_status_from_image(image_path, False)
    
    with open(query_path, 'r') as file:
        queries = file.readlines()
    queries = [query.strip() for query in queries if query.strip()]

    for i in range(1, len(queries)):
        status = parking_spaces[int(queries[i]) - 1] if queries[i].isdigit() else False
        status = 1 if status else 0
        queries[i] += f" {status}"

    return "".join(queries)

def task_to_results(task_name: str, path_to_tasks: str, save_to_folder: str, remove_old_results=False) -> None:
    """
    ======= General function to run tasks and save results =======

    This function runs the specified task and saves the results to a specified folder.
    """
    # Validate inputs
    task_name = task_name.strip().lower()
    assert task_name in ["task1", "task2", "task3", "task4"], f"Invalid task name: {task_name}"

    # Make preparations for saving results
    if not os.path.exists(save_to_folder):
        os.makedirs(save_to_folder)
        
    if remove_old_results:
        for file in os.listdir(save_to_folder):
            if file.endswith("_results.txt"):
                os.remove(os.path.join(save_to_folder, file))
    path_to_task_folder = os.path.join(path_to_tasks, task_name.capitalize())

    # List all files in the dataset folder and create a dictionary to map file names to assets
    files = os.listdir(path_to_task_folder)
    files = [file for file in files if file.endswith((".jpg", ".mp4", ".txt"))] 
    files_dict = {}
    for file in files:
        path_to_file = os.path.join(path_to_task_folder, file)
        extension = file.split(".")[-1]
        file_name = file.split(".")[0]
        if file_name.count("_") > 1:
            file_name = "_".join(file_name.split("_")[:2])

        if file_name not in files_dict:
            files_dict[file_name] = {"mp4": None, "jpg": None, "txt": None}
        files_dict[file_name][extension] = path_to_file

    # Go through each file and run the task
    if RUN_SELECTED_IMAGE: # Run only on the selected file
        files_dict = {k: v for k, v in files_dict.items() if k == SELECTED_FILE.split(".")[0]}

    for file_name in files_dict:
        print (f"Found file: {file_name}")
        print (f"File: {files_dict[file_name]}")
        mp4_file = files_dict[file_name].get("mp4")
        jpg_file = files_dict[file_name].get("jpg")
        txt_file = files_dict[file_name].get("txt")

        test_file = mp4_file if mp4_file else jpg_file
        if not test_file:
            logger.warning(f"No valid file found for {file_name}. Skipping.")
            continue
        
        # Call function dynamically based on the task name        
        result = None
        if task_name in globals():
            callable_func = globals()[task_name]
            if callable_func:
                logger.info(f"Running {task_name} for {file_name}.")
                n_arguments = callable_func.__code__.co_argcount
                if n_arguments == 1:
                    result = callable_func(test_file)
                elif n_arguments == 2 and txt_file:
                    result = callable_func(test_file, txt_file)
            else:
                logger.error(f"Function {task_name} not found.")

        if not SAVE_TO_FOLDER_SWITCH:
            logger.info(f"Results for {file_name}: {result}")
            continue

        # Save results to the specified folder
        if result is not None:
            results_file = os.path.join(save_to_folder, f"{file_name}_results.txt")
            with open(results_file, 'w') as file:
                file.write(f"{result}\n")
        else:
            logger.warning(f"No results found for {file_name}.")

def main() -> int:
    """ ======= Main function to run the tasks of the project ======= """
    path_to_tasks = "train"
    task_name = "Task4"
    
    save_to_folder = f"train/{task_name}/results"
    task_to_results(task_name, path_to_tasks, save_to_folder, remove_old_results=True)

    # task2("train/Task2/01.mp4")

    # parking_spaces = get_parking_spaces_status_from_image(SELECTED_IMAGE, True)
    # for i in range(len(parking_spaces) -1, -1, -1):
    #     logger.info(f"Parking space {i + 1}: {'Occupied' if parking_spaces[i] else 'Free'}")


if __name__ == "__main__":
    exit(main())
