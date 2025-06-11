from config import *
from image_utils import ImageUtils
from utils import Point, Quadrilateral, get_logger
import cv2
import numpy as np

logger = get_logger("ImageProcessor")

class ImageProcessor:
    """ ======= Class to handle complex image processing tasks ======="""

    # The coordinates for the parking spaces quadrilateral in the image

    PARKING_BOX = Quadrilateral(
        top_left=Point(410, 230),
        bottom_right=Point(915, 500),
        top_right=Point(455, 210),
        bottom_left=Point(915, 600)
    )
    
    TRAFFIC_LIGHT_QUEUE = Quadrilateral(
        top_left=Point(410, 230),
        bottom_right=Point(915, 500),
        top_right=Point(455, 210),
        bottom_left=Point(915, 600)
    )

    # The y-coordinates of the parking lines in the image
    # These lines are used to determine the parking spaces
    # The values are based on the image after perspective transformation based on above points
    PARKING_LINES = [10, 100, 192, 290, 385, 480, 580, 680, 785, 890] 

    @staticmethod
    def get_image_with_cars_from_quadrilateral(image: np.ndarray, quadrilateral: Quadrilateral, 
                                               car_points: list[Point], display_intermediate:bool=False) -> np.ndarray:
        """Get an image cropped to the specified quadrilateral together with the car points."""
        # ======= Filtering car points outside the target quadrilateral =======
        logger.trace("Filtering car points outside the target quadrilateral.")
        car_points_inside_quadrilateral = []
        car_points.sort(key=lambda p: p.y)
        for point in car_points:
            if quadrilateral.check_point_inside(point):
                car_points_inside_quadrilateral.append(point)
                logger.debug(f"{point} is inside the target quadrilateral.")
            else:
                logger.debug(f"{point} is outside the target quadrilateral, removing it.")

        image = ImageUtils.load_image(SELECTED_IMAGE)
        image = ImageUtils.resize_with_aspect_ratio(image, width=1000)

        # ======= Warp and crop the image to the target quadrilateral =======
        logger.trace("Applying perspective transformation to the image.")
        src_points = [   
            quadrilateral.bottom_left, quadrilateral.bottom_right,
            quadrilateral.top_right, quadrilateral.top_left
        ]
        warped_image, warp_matrix = ImageUtils.warp_perspective(image, src_points)
        ImageUtils.display(warped_image, title="Warped Image", display=display_intermediate)

        # ======= Apply warp matrix to car points =======
        logger.trace("Warping car points to the warped image coordinates.")
        warped_car_points = []
        for point in car_points_inside_quadrilateral:
            warped_point = ImageUtils.warp_point_using_matrix(point, warp_matrix)
            warped_car_points.append(warped_point)
            
        # ======= Draw and display the car points in the warped image of the quadrilateral =======
        if display_intermediate:
            logger.trace("Drawing warped car points on the warped image.")
            edges_warped_image = ImageUtils.get_edges(warped_image)
            edges_warped_image = cv2.cvtColor(edges_warped_image, cv2.COLOR_GRAY2BGR)
            for point in warped_car_points:
                ImageUtils.draw_point_on_image(edges_warped_image, point)
            ImageUtils.display(edges_warped_image, title="Edges of Warped Image", display=display_intermediate)
        
        return warped_image, warped_car_points
    
    @staticmethod
    def check_parking_spaces(image: np.ndarray, car_points: list[Point], display_intermediate=False) -> list[bool]:
        """Check if parking spaces availability based on car points."""
        logger.trace("Starting parking spaces availability check.")

        _, warped_car_points = ImageProcessor.get_image_with_cars_from_quadrilateral(
            image, ImageProcessor.PARKING_BOX, car_points, display_intermediate
        )

        # ======= Check if the warped car points are within the parking spaces defined by PARKING_LINES =======
        logger.trace("Checking parking spaces based on warped car points.")
        parking_spaces = []
        for i, line_y in enumerate(ImageProcessor.PARKING_LINES):
            next_line_y = ImageProcessor.PARKING_LINES[i + 1] if i + 1 < len(ImageProcessor.PARKING_LINES) else 1000

            occupied = False
            for point in warped_car_points:
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
    # for i, space in enumerate(parking_spaces):
    #     logger.info(f"Parking space {i + 1}: {'Occupied' if space else 'Free'}")

    # image = ImageUtils.load_image(SELECTED_IMAGE)
    # image = ImageUtils.resize_with_aspect_ratio(image, width=1000)
    # ImageUtils.draw_quadrilateral_on_image(image, ImageProcessor.PARKING_BOX)
    # ImageUtils.display(image, title="Parking Spaces")

    return 0

if __name__ == "__main__":
    exit(main())
