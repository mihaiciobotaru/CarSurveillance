from config import *
from utils import Point, Box, Rectangle, Line, get_logger
import cv2
import numpy as np

logger = get_logger("ImageUtils")

class ImageUtils:
    """ ======= Class to handle simple and more common image processing tasks ======="""
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate an image by a given angle."""
        if len(image.shape) == 3:
            h, w = image.shape[:2]
        else:
            h, w = image.shape
        
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        return rotated_image, rotation_matrix
    
    @staticmethod
    def warp_perspective(image: np.ndarray, src_points: list[Point], dst_points: list[Point]=[]) -> np.ndarray:
        """Apply perspective transformation to an image."""
        if len(src_points) != 4:
            raise ValueError("src_points must contain exactly 4 points.")
        
        if len(dst_points) == 0:
            # Default destination points for a top-down view
            dst_points = [
                Point(0, 0), Point(1000, 0), 
                Point(0, 1000), Point(1000, 1000)
            ]
        elif len(dst_points) != 4:
            raise ValueError("dst_points must contain exactly 4 points, be empty or left unset for default values.")

        src_np = np.float32([point.to_tuple() for point in src_points])
        dst_np = np.float32([point.to_tuple() for point in dst_points])
        
        warp_matrix = cv2.getPerspectiveTransform(src_np, dst_np)
        warped_image = cv2.warpPerspective(image, warp_matrix, (image.shape[1], image.shape[0]))
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
        image = cv2.imread(image_path)
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
    def display(image: np.ndarray|str, title: str = "Image", display=True, size: tuple[int, int]|int = 1000) -> None:
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
                image = ImageUtils.resize_with_aspect_ratio(image, width=size)
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
    def draw_box_on_image(image: np.ndarray, box: Box) -> np.ndarray:
        """Draw a box on the image."""
        ImageUtils.draw_line_on_image(image, Line(box.top_left, box.top_right))
        ImageUtils.draw_line_on_image(image, Line(box.top_right, box.bottom_right))
        ImageUtils.draw_line_on_image(image, Line(box.bottom_right, box.bottom_left))
        ImageUtils.draw_line_on_image(image, Line(box.bottom_left, box.top_left))
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
