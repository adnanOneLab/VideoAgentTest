"""
Blueprint Processing for CCTV-Store Mapping

This code converts a mall blueprint image with marked cameras into a spatial data format
that can be used by the CCTV-Store mapping system.

Steps:
1. Process the blueprint image to establish a coordinate system
2. Extract camera positions and orientations
3. Define store boundaries
4. Generate the JSON representation of the spatial data
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import os
import math

class BlueprintProcessor:
    def __init__(self):
        self.image = None
        self.scale_factor = 1.0  # meters per pixel
        self.origin = (0, 0)  # (x, y) coordinate of origin in pixels
        self.cameras = {}
        self.stores = {}
        self.floor = 1  # Default floor number
        
    def load_blueprint(self, image_path):
        """Load the blueprint image"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to RGB for display
        self.display_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        print(f"Blueprint loaded. Dimensions: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        return True
    
    def set_scale(self, pixels, meters):
        """Set the scale factor (meters per pixel)"""
        self.scale_factor = meters / pixels
        print(f"Scale set to {self.scale_factor:.4f} meters per pixel")
        return True
    
    def set_origin(self, x, y):
        """Set the origin point (in pixels)"""
        self.origin = (x, y)
        print(f"Origin set to pixel coordinates ({x}, {y})")
        return True
    
    def set_floor(self, floor_number):
        """Set the floor number for the current blueprint"""
        self.floor = floor_number
        print(f"Floor set to {floor_number}")
        return True
    
    def pixels_to_meters(self, px, py):
        """Convert pixel coordinates to meter coordinates"""
        # Calculate relative to origin
        rel_x = px - self.origin[0]
        rel_y = self.origin[1] - py  # Invert y-axis to match standard coordinate system
        
        # Convert to meters
        meter_x = rel_x * self.scale_factor
        meter_y = rel_y * self.scale_factor
        
        return meter_x, meter_y
    
    def meters_to_pixels(self, mx, my):
        """Convert meter coordinates to pixel coordinates"""
        # Convert to pixels
        rel_x = mx / self.scale_factor
        rel_y = my / self.scale_factor
        
        # Adjust relative to origin
        px = int(self.origin[0] + rel_x)
        py = int(self.origin[1] - rel_y)  # Invert y-axis
        
        return px, py
    
    def add_camera_manual(self, camera_id, pixel_x, pixel_y, orientation, fov_angle=70, fov_range=20):
        """
        Add a camera manually by specifying its position and orientation
        - camera_id: unique identifier for the camera
        - pixel_x, pixel_y: position in pixel coordinates
        - orientation: camera direction in degrees (0 is east, 90 is north, etc.)
        - fov_angle: field of view angle in degrees
        - fov_range: maximum viewing distance in meters
        """
        # Convert pixel position to meter coordinates
        meter_x, meter_y = self.pixels_to_meters(pixel_x, pixel_y)
        
        self.cameras[camera_id] = {
            "camera_id": camera_id,
            "location": {
                "x": meter_x,
                "y": meter_y,
                "floor": self.floor,
                "orientation": orientation
            },
            "field_of_view": {
                "angle": fov_angle,
                "range": fov_range
            },
            "resolution": {
                "width": 1920,  # Default resolution
                "height": 1080
            },
            "status": "active"
        }
        
        print(f"Camera {camera_id} added at position ({meter_x:.2f}, {meter_y:.2f}) meters with orientation {orientation}°")
        return True
    
    def detect_cameras_by_color(self, lower_color, upper_color, min_area=100):
        """
        Automatically detect cameras based on color markers
        - lower_color, upper_color: HSV color range for camera markers
        - min_area: minimum area for a camera marker
        """
        # Convert image to HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Create mask for the specified color range
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        camera_count = 0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area >= min_area:
                # Get center of contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Create camera ID
                    camera_id = f"cam{camera_count+1:03d}"
                    
                    # Default orientation (can be adjusted later)
                    orientation = 0
                    
                    # Add camera
                    self.add_camera_manual(camera_id, cx, cy, orientation)
                    camera_count += 1
        
        print(f"Detected {camera_count} cameras based on color markers")
        return camera_count
    
    def add_store_manual(self, store_id, name, category, pixel_polygon, metadata=None):
        """
        Add a store manually by specifying its boundary polygon
        - store_id: unique identifier for the store
        - name: store name
        - category: store category
        - pixel_polygon: list of (x,y) pixel coordinates defining the store boundaries
        - metadata: optional additional store information
        """
        # Convert pixel polygon to meter coordinates
        meter_polygon = []
        for px, py in pixel_polygon:
            mx, my = self.pixels_to_meters(px, py)
            meter_polygon.append({"x": mx, "y": my})
        
        self.stores[store_id] = {
            "store_id": store_id,
            "name": name,
            "category": category,
            "location": {
                "floor": self.floor,
                "polygon": meter_polygon
            },
            "metadata": metadata or {}
        }
        
        print(f"Store {store_id} '{name}' added with {len(pixel_polygon)} boundary points")
        return True
    
    def store_selection_tool(self):
        """
        Interactive tool to define store boundaries by clicking on the blueprint
        """
        # Create a copy of the image for display
        display_img = self.display_image.copy()
        
        # State variables
        store_id = None
        store_name = None
        store_category = None
        points = []
        
        # Function to handle mouse clicks
        def click_event(event, x, y, flags, params):
            nonlocal points, display_img
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # Add point to the list
                points.append((x, y))
                
                # Draw point on the image
                cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
                
                # If we have at least 2 points, draw a line
                if len(points) > 1:
                    cv2.line(display_img, points[-2], points[-1], (0, 255, 0), 2)
                
                # Display the updated image
                cv2.imshow("Store Selection", display_img)
                
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Complete the polygon if we have at least 3 points
                if len(points) >= 3:
                    # Draw the closing line
                    cv2.line(display_img, points[-1], points[0], (0, 255, 0), 2)
                    cv2.imshow("Store Selection", display_img)
                    
                    # Save the store
                    if store_id and store_name and store_category:
                        self.add_store_manual(store_id, store_name, store_category, points)
                    
                    # Reset for next store
                    points.clear()
                    display_img = self.display_image.copy()
                    
                    # Draw existing cameras
                    self.draw_cameras(display_img)
                    
                    # Draw existing stores
                    self.draw_stores(display_img)
                    
                    # Display the updated image
                    cv2.imshow("Store Selection", display_img)
        
        # Get store information
        store_id = input("Enter store ID (e.g., store001): ")
        store_name = input("Enter store name: ")
        store_category = input("Enter store category: ")
        
        # Create a window and set the callback
        cv2.namedWindow("Store Selection")
        cv2.setMouseCallback("Store Selection", click_event)
        
        # Draw existing cameras
        self.draw_cameras(display_img)
        
        # Draw existing stores
        self.draw_stores(display_img)
        
        # Show the image
        cv2.imshow("Store Selection", display_img)
        
        print("Click to add points, right-click to complete the polygon. Press 'q' to exit.")
        
        # Wait for key press
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                # Start a new store
                points.clear()
                display_img = self.display_image.copy()
                
                # Draw existing cameras
                self.draw_cameras(display_img)
                
                # Draw existing stores
                self.draw_stores(display_img)
                
                # Display the updated image
                cv2.imshow("Store Selection", display_img)
                
                # Get new store information
                store_id = input("Enter store ID (e.g., store001): ")
                store_name = input("Enter store name: ")
                store_category = input("Enter store category: ")
        
        cv2.destroyAllWindows()
        return True
    
    def camera_orientation_tool(self):
        """
        Interactive tool to set camera orientations by dragging from camera position
        """
        # Create a copy of the image for display
        display_img = self.display_image.copy()
        
        # State variables
        selected_camera = None
        start_point = None
        
        # Draw existing cameras and stores
        self.draw_cameras(display_img)
        self.draw_stores(display_img)
        
        # Function to handle mouse events
        def click_event(event, x, y, flags, params):
            nonlocal selected_camera, start_point, display_img
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if the click is near a camera
                for camera_id, camera in self.cameras.items():
                    px, py = self.meters_to_pixels(camera["location"]["x"], camera["location"]["y"])
                    distance = math.sqrt((x - px)**2 + (y - py)**2)
                    
                    if distance < 10:  # 10-pixel radius for selection
                        selected_camera = camera_id
                        start_point = (px, py)
                        break
            
            elif event == cv2.EVENT_MOUSEMOVE and selected_camera and start_point:
                # Create a fresh copy to draw on
                temp_img = self.display_image.copy()
                self.draw_cameras(temp_img)
                self.draw_stores(temp_img)
                
                # Draw a line from camera to current mouse position
                cv2.line(temp_img, start_point, (x, y), (0, 0, 255), 2)
                
                # Show the updated image
                cv2.imshow("Camera Orientation", temp_img)
            
            elif event == cv2.EVENT_LBUTTONUP and selected_camera and start_point:
                # Calculate orientation
                dx = x - start_point[0]
                dy = start_point[1] - y  # Invert y since pixels increase downward
                
                orientation = math.degrees(math.atan2(dy, dx))
                # Normalize to 0-360 range
                orientation = (orientation + 360) % 360
                
                # Update camera orientation
                self.cameras[selected_camera]["location"]["orientation"] = orientation
                
                print(f"Camera {selected_camera} orientation set to {orientation:.1f}°")
                
                # Reset state
                selected_camera = None
                start_point = None
                
                # Redraw with updated camera orientations
                display_img = self.display_image.copy()
                self.draw_cameras(display_img)
                self.draw_stores(display_img)
                cv2.imshow("Camera Orientation", display_img)
        
        # Create a window and set the callback
        cv2.namedWindow("Camera Orientation")
        cv2.setMouseCallback("Camera Orientation", click_event)
        
        # Show the image
        cv2.imshow("Camera Orientation", display_img)
        
        print("Click and drag from camera position to set orientation. Press 'q' to exit.")
        
        # Wait for key press
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return True
    
    def draw_cameras(self, img):
        """Draw cameras on the image"""
        for camera_id, camera in self.cameras.items():
            # Convert meter coordinates to pixels
            px, py = self.meters_to_pixels(camera["location"]["x"], camera["location"]["y"])
            
            # Draw camera marker
            cv2.circle(img, (px, py), 5, (255, 0, 0), -1)
            
            # Draw camera ID
            cv2.putText(img, camera_id, (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw field of view cone
            orientation = camera["location"]["orientation"]
            fov_angle = camera["field_of_view"]["angle"]
            fov_range_meters = camera["field_of_view"]["range"]
            
            # Convert range to pixels
            fov_range_pixels = int(fov_range_meters / self.scale_factor)
            
            # Calculate cone edges
            half_angle = fov_angle / 2
            angle1 = math.radians(orientation - half_angle)
            angle2 = math.radians(orientation + half_angle)
            
            # Calculate end points of cone edges
            edge1_x = px + int(fov_range_pixels * math.cos(angle1))
            edge1_y = py - int(fov_range_pixels * math.sin(angle1))  # Minus for y-coordinate
            
            edge2_x = px + int(fov_range_pixels * math.cos(angle2))
            edge2_y = py - int(fov_range_pixels * math.sin(angle2))  # Minus for y-coordinate
            
            # Draw cone
            cv2.line(img, (px, py), (edge1_x, edge1_y), (255, 0, 0), 1)
            cv2.line(img, (px, py), (edge2_x, edge2_y), (255, 0, 0), 1)
            
            # Draw arc
            cv2.ellipse(img, (px, py), (fov_range_pixels, fov_range_pixels), 
                      0, -orientation + half_angle, -orientation - half_angle, (255, 0, 0), 1)
    
    def draw_stores(self, img):
        """Draw stores on the image"""
        for store_id, store in self.stores.items():
            # Convert meter coordinates to pixels
            pixel_polygon = []
            for point in store["location"]["polygon"]:
                px, py = self.meters_to_pixels(point["x"], point["y"])
                pixel_polygon.append((px, py))
            
            # Convert to numpy array
            pixel_polygon = np.array(pixel_polygon, np.int32)
            pixel_polygon = pixel_polygon.reshape((-1, 1, 2))
            
            # Draw store polygon
            cv2.polylines(img, [pixel_polygon], True, (0, 255, 0), 2)
            
            # Find centroid for label
            centroid_x = np.mean([p[0] for p in pixel_polygon.reshape(-1, 2)])
            centroid_y = np.mean([p[1] for p in pixel_polygon.reshape(-1, 2)])
            
            # Draw store name
            cv2.putText(img, store["name"], (int(centroid_x), int(centroid_y)), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def visualize_map(self):
        """Visualize the current map with cameras and stores"""
        # Create a fresh copy of the image
        display_img = self.display_image.copy()
        
        # Draw cameras
        self.draw_cameras(display_img)
        
        # Draw stores
        self.draw_stores(display_img)
        
        # Show the image
        plt.figure(figsize=(12, 10))
        plt.imshow(display_img)
        plt.title('Mall Blueprint with Cameras and Stores')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return display_img
    
    def export_to_json(self, output_file):
        """Export the spatial data to a JSON file"""
        # Create data structure
        data = {
            "mall_info": {
                "name": "Mall Blueprint",
                "floors": [self.floor],
                "scale_factor": self.scale_factor,
                "origin": {"x": self.origin[0], "y": self.origin[1]}
            },
            "cameras": list(self.cameras.values()),
            "stores": list(self.stores.values()),
            "reference_points": []  # Can be populated if needed
        }
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported data to {output_file}")
        return True
    
    def import_from_json(self, input_file):
        """Import spatial data from a JSON file"""
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Load mall info
        if "mall_info" in data:
            mall_info = data["mall_info"]
            if "scale_factor" in mall_info:
                self.scale_factor = mall_info["scale_factor"]
            if "origin" in mall_info:
                self.origin = (mall_info["origin"]["x"], mall_info["origin"]["y"])
            if "floors" in mall_info and len(mall_info["floors"]) > 0:
                self.floor = mall_info["floors"][0]
        
        # Load cameras
        if "cameras" in data:
            self.cameras = {camera["camera_id"]: camera for camera in data["cameras"]}
        
        # Load stores
        if "stores" in data:
            self.stores = {store["store_id"]: store for store in data["stores"]}
        
        print(f"Imported data from {input_file}")
        print(f"Loaded {len(self.cameras)} cameras and {len(self.stores)} stores")
        return True


# =========================================
# USAGE EXAMPLE
# =========================================

def example_process_blueprint():
    # Create processor
    processor = BlueprintProcessor()
    
    # Load blueprint (would be your actual blueprint image)
    processor.load_blueprint("mall_blueprint.jpg")
    
    # Set scale - example: 100 pixels = 10 meters
    processor.set_scale(100, 10)
    
    # Set origin (bottom-left corner of the image)
    height, width = processor.image.shape[:2]
    processor.set_origin(0, height)
    
    # Add cameras manually
    processor.add_camera_manual("cam001", 100, 200, 45, 70, 20)
    processor.add_camera_manual("cam002", 300, 150, 270, 70, 20)
    
    # Add stores manually
    store_polygon = [(50, 100), (150, 100), (150, 250), (50, 250)]
    processor.add_store_manual("store001", "Fashion Outlet", "Clothing", store_polygon)
    
    # Visualize the map
    processor.visualize_map()
    
    # Export to JSON
    processor.export_to_json("mall_spatial_data.json")
    
    print("Blueprint processing complete!")
    return True


def interactive_blueprint_processing():
    # Create processor
    processor = BlueprintProcessor()
    
    # User interaction
    print("===== Mall Blueprint Processing Tool =====")
    
    # Load blueprint
    blueprint_path = input("Enter path to blueprint image: ")
    processor.load_blueprint(blueprint_path)
    
    # Set scale
    pixels = float(input("Enter pixel distance for scale reference: "))
    meters = float(input("Enter corresponding distance in meters: "))
    processor.set_scale(pixels, meters)
    
    # Set origin
    height, width = processor.image.shape[:2]
    x_origin = float(input(f"Enter x-coordinate for origin (0-{width}): ") or 0)
    y_origin = float(input(f"Enter y-coordinate for origin (0-{height}): ") or height)
    processor.set_origin(x_origin, y_origin)
    
    # Set floor
    floor = int(input("Enter floor number: ") or 1)
    processor.set_floor(floor)
    
    # Menu loop
    while True:
        print("\nOptions:")
        print("1. Add camera manually")
        print("2. Add store manually")
        print("3. Use interactive store selection tool")
        print("4. Use camera orientation tool")
        print("5. Visualize current map")
        print("6. Export to JSON")
        print("7. Import from JSON")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ")
        
        if choice == '1':
            # Add camera manually
            camera_id = input("Enter camera ID: ")
            x = float(input("Enter x-coordinate (pixels): "))
            y = float(input("Enter y-coordinate (pixels): "))
            orientation = float(input("Enter orientation (degrees, 0-360): "))
            fov_angle = float(input("Enter field of view angle (degrees): ") or "70")
            fov_range = float(input("Enter field of view range (meters): ") or "20")
            processor.add_camera_manual(camera_id, x, y, orientation, fov_angle, fov_range)
        
        elif choice == '2':
            # Add store manually
            store_id = input("Enter store ID: ")
            name = input("Enter store name: ")
            category = input("Enter store category: ")
            
            print("Enter polygon points (x,y) one by one. Enter 'done' when finished.")
            polygon = []
            while True:
                point = input("Enter point (x,y) or 'done': ")
                if point.lower() == 'done':
                    break
                x, y = map(float, point.split(','))
                polygon.append((x, y))
            
            processor.add_store_manual(store_id, name, category, polygon)
        
        elif choice == '3':
            # Use interactive store selection tool
            processor.store_selection_tool()
        
        elif choice == '4':
            # Use camera orientation tool
            processor.camera_orientation_tool()
        
        elif choice == '5':
            # Visualize current map
            processor.visualize_map()
        
        elif choice == '6':
            # Export to JSON
            output_file = input("Enter output file path: ")
            processor.export_to_json(output_file)
        
        elif choice == '7':
            # Import from JSON
            input_file = input("Enter input file path: ")
            processor.import_from_json(input_file)
        
        elif choice == '8':
            # Exit
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")
    
    return True

# Run the interactive tool if script is executed directly
if __name__ == "__main__":
    interactive_blueprint_processing()