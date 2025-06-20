from config import *
from utils import Point, Quadrilateral, Rectangle, Line, get_logger
import cv2
import numpy as np

logger = get_logger("ImageUtils")

class ImageUtils:
    """ ======= Class to handle simple and more common image processing tasks ======="""
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate an image by a given angle."""
        if len(image.shape) == 3:
            height, width = image.shape[:2]
        else:
            height, width = image.shape

        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])

        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        return rotated_image, rotation_matrix
    
    @staticmethod
    def crop_image_to_quadrilateral(image: np.ndarray, quadrilateral: Quadrilateral) -> np.ndarray:
        """Crop an image to a specified quadrilateral."""
        bounding_box = quadrilateral.get_bounding_box()
        min_x = bounding_box.top_left.x
        max_x = bounding_box.bottom_right.x
        min_y = bounding_box.top_left.y
        max_y = bounding_box.bottom_right.y
        if min_x < 0 or min_y < 0 or max_x > image.shape[1] or max_y > image.shape[0]:
            raise ValueError("Quadrilateral points are out of image bounds.")

        cropped_image = image[min_y:max_y, min_x:max_x]
        if cropped_image.size == 0:
            raise ValueError("Cropped image is empty. Check the quadrilateral points.")
        return cropped_image
    
    @staticmethod
    def warp_perspective(image: np.ndarray, src_points: list[Point], dst_points: list[Point]=[]) -> np.ndarray:
        """Apply perspective transformation to an image."""
        if len(src_points) != 4:
            raise ValueError("src_points must contain exactly 4 points.")
        
        if len(dst_points) == 0:
            # Default destination points for a top-down view
            dst_points = [
                Point(1000, 1000), Point(0, 1000), 
                Point(0, 0), Point(1000, 0)
            ]
        elif len(dst_points) != 4:
            raise ValueError("dst_points must contain exactly 4 points, be empty or left unset for default values.")
        
        src_np = np.float32([point.to_tuple() for point in src_points])
        dst_np = np.float32([point.to_tuple() for point in dst_points])

        warp_matrix = cv2.getPerspectiveTransform(src_np, dst_np)
        warped_image = cv2.warpPerspective(image, warp_matrix, (1000, 1000))
        return warped_image, warp_matrix
    
    @staticmethod
    def warp_point_using_matrix(point: Point, warp_matrix: np.ndarray) -> Point:
        """Warp a point using a given perspective transformation matrix."""
        point_np = np.float32([[[point.x, point.y]]])
        warped_point = cv2.perspectiveTransform(point_np, warp_matrix)
        return Point(warped_point[0][0][0], warped_point[0][0][1])

    @staticmethod
    def rotate_point_using_matrix(point: Point, rotation_matrix: np.ndarray) -> Point:
        """Rotate a point using a given rotation matrix."""
        point_np = np.float32([[[point.x, point.y]]])
        rotated_point = cv2.transform(point_np, rotation_matrix)
        return Point(rotated_point[0][0][0], rotated_point[0][0][1])
    
    @staticmethod
    def get_edges(image: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection to an image."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image, 50, 500)
        return edges

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load an image from the specified path."""
        if isinstance(image_path, np.ndarray):
            logger.warning(f"Expected image path, but received an image array. Returning the image as is.")
            return image_path

        if not isinstance(image_path, str):
            raise ValueError(f"Expected image path as a string, but received {type(image_path)}.")

        try:
            image = cv2.imread(image_path)
        except Exception as e:
            raise ValueError(f"Error reading image from {image_path}: {e}")
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        return image
    
    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray, width: int=None, height: int=None):
        """
        Resizes an image while maintaining its aspect ratio.

        Args:
            image: The input image.
            width: The desired width of the resized image.
            height: The desired height of the resized image.

        Returns:
            The resized image.
        """
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def display(image: np.ndarray|str, title: str = "Image", display=True, size: tuple[int, int]|int = 900) -> None:
        """Display an image in a window."""
        
        if display is False:
            return
        
        if isinstance(image, str): # image variable is a path to an image file; we need to load it
            try:
                image = ImageUtils.load_image(image)
            except ValueError as e:
                print(e)
                return

        try:
            if isinstance(size, int):
                height, width = image.shape[:2]
                if width > height:
                    image = ImageUtils.resize_with_aspect_ratio(image, width=size)
                else:
                    image = ImageUtils.resize_with_aspect_ratio(image, height=size)
            elif isinstance(size, tuple) and len(size) == 2:
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            
            cv2.imshow(title, image)
            while True: # pressing ESC or closing the window will exit the loop and stop displaying the image
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error displaying image: {e}")
            return

    @staticmethod
    def draw_line_on_image(image: np.ndarray, line: Line) -> np.ndarray:
        """Draw a line on the image."""
        if line.start and line.end:
            cv2.line(image, line.start.to_tuple(), line.end.to_tuple(), (0, 255, 0), 2)
        return image
    
    @staticmethod  
    def draw_quadrilateral_on_image(image: np.ndarray, quadrilateral: Quadrilateral) -> np.ndarray:
        """Draw a quadrilateral on the image."""
        ImageUtils.draw_line_on_image(image, Line(quadrilateral.top_left, quadrilateral.top_right))
        ImageUtils.draw_line_on_image(image, Line(quadrilateral.top_right, quadrilateral.bottom_right))
        ImageUtils.draw_line_on_image(image, Line(quadrilateral.bottom_right, quadrilateral.bottom_left))
        ImageUtils.draw_line_on_image(image, Line(quadrilateral.bottom_left, quadrilateral.top_left))
        return image
    
    @staticmethod
    def draw_rectangle_on_image(image: np.ndarray, rectangle: Rectangle, text:str='') -> np.ndarray:
        """Draw a rectangle on the image."""
        cv2.rectangle(image, rectangle.top_left.to_tuple(), rectangle.bottom_right.to_tuple(), (0, 255, 0), 2)
        if len(text) > 0:
            cv2.putText(image, text, (rectangle.top_left.x + 10, rectangle.top_left.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    @staticmethod
    def draw_point_on_image(image: np.ndarray, point: Point, text:str='') -> np.ndarray:
        """Draw a point on the image."""
        cv2.circle(image, point.to_tuple(), 5, (0, 255, 0), -1)
        if len(text) > 0:
            cv2.putText(image, text, (point.x + 10, point.y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image


def main() -> int:
    """Main function to demonstrate the ImageUtils class functionality."""
    ImageUtils.display(SELECTED_IMAGE, "Example Image")

    return 0

if __name__ == "__main__":
    exit(main())
