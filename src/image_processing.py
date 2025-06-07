from config import *
from image_utils import ImageUtils
from utils import Point, Box, get_logger
import cv2
import numpy as np

logger = get_logger("ImageProcessor")

class ImageProcessor:
    """ ======= Class to handle complex image processing tasks ======="""

    # The coordinates for the parking spaces quadrilateral in the image
    TOP_RIGHT = Point(455, 210)
    BOTTOM_RIGHT = Point(915, 500)
    TOP_LEFT = Point(410, 230)
    BOTTOM_LEFT = Point(915, 600)

    PARKING_BOX = Box(
        top_left=TOP_LEFT,
        bottom_right=BOTTOM_RIGHT,
        top_right=TOP_RIGHT,
        bottom_left=BOTTOM_LEFT
    )

    # The y-coordinates of the parking lines in the image
    # These lines are used to determine the parking spaces
    # The values are based on the image after perspective transformation based on above points
    PARKING_LINES = [10, 100, 192, 290, 385, 480, 580, 680, 785, 890] 

    @staticmethod
    def check_parking_spaces(image: np.ndarray, car_points: list[Point], display_intermediate=False) -> list[bool]:
        """Check if parking spaces availability based on car points."""
        logger.trace("Starting parking spaces availability check.")

        # ======= Filtering car points outside the parking box =======
        logger.trace("Filtering car points outside the parking box.")
        car_points_inside_box = []
        car_points.sort(key=lambda p: p.y)
        for point in car_points:
            if ImageProcessor.PARKING_BOX.check_point_inside(point):
                car_points_inside_box.append(point)
                logger.debug(f"{point} is inside the parking box.")
            else:
                logger.debug(f"{point} is outside the parking box, removing it.")

        image = ImageUtils.load_image(SELECTED_IMAGE)
        image = ImageUtils.resize_with_aspect_ratio(image, width=1000)

        # ======= Warp and crop the image to get a top-down view of the parking spaces =======
        logger.trace("Applying perspective transformation to the image.")
        src_points = [
            ImageProcessor.TOP_RIGHT, ImageProcessor.BOTTOM_RIGHT,
            ImageProcessor.TOP_LEFT, ImageProcessor.BOTTOM_LEFT
        ]
        warped_image, warp_matrix = ImageUtils.warp_perspective(image, src_points)
        rotated_warped_image, rotation_matrix = ImageUtils.rotate(warped_image, angle=-90)
        ImageUtils.display(rotated_warped_image, title="Rotated Warped Image", display=display_intermediate)
        
        # ======= Apply warp matrix to car points =======
        logger.trace("Translating car points to the warped image coordinates.")
        translated_car_points = []
        for point in car_points_inside_box:
            warped_point = ImageUtils.warp_point_using_matrix(point, warp_matrix)
            translated_point = ImageUtils.rotate_point_using_matrix(warped_point, rotation_matrix)
            translated_car_points.append(translated_point)
            
        # ======= Draw and display the car points in the warped image of parking spaces =======
        if display_intermediate:
            logger.trace("Drawing translated car points on the rotated warped image.")
            edges_rotated_warped_image = ImageUtils.get_edges(rotated_warped_image)
            edges_rotated_warped_image = cv2.cvtColor(edges_rotated_warped_image, cv2.COLOR_GRAY2BGR)
            for point in translated_car_points:
                ImageUtils.draw_point_on_image(edges_rotated_warped_image, point)
            ImageUtils.display(edges_rotated_warped_image, title="Edges of Rotated Warped Image", display=display_intermediate)

        # ======= Check if the translated car points are within the parking spaces defined by PARKING_LINES =======
        logger.trace("Checking parking spaces based on translated car points.")
        parking_spaces = []
        for i, line_y in enumerate(ImageProcessor.PARKING_LINES):
            next_line_y = ImageProcessor.PARKING_LINES[i + 1] if i + 1 < len(ImageProcessor.PARKING_LINES) else 1000

            occupied = False
            for point in translated_car_points:
                if line_y <= point.y <= next_line_y:
                    occupied = True
                    break
            parking_spaces.append(occupied)
        
        parking_spaces.reverse()
        return parking_spaces

def main() -> int:
    """ ======= Debug case for parking spaces status detection ======= """
    logger = get_logger("Main")
    cars = [Point(540, 290), Point(694, 405)]
    parking_spaces = ImageProcessor.check_parking_spaces(SELECTED_IMAGE, cars, display_intermediate=DISPLAY_PARKING_SPACES_INTERMEDIATE)
    for i, space in enumerate(parking_spaces):
        logger.info(f"Parking space {i + 1}: {'Occupied' if space else 'Free'}")

    image = ImageUtils.load_image(SELECTED_IMAGE)
    image = ImageUtils.resize_with_aspect_ratio(image, width=1000)
    ImageUtils.draw_box_on_image(image, ImageProcessor.PARKING_BOX)
    ImageUtils.display(image, title="Parking Spaces")

    return 0

if __name__ == "__main__":
    exit(main())
