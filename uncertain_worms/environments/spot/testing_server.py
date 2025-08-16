# type:ignore
import os

import cv2
import numpy as np
import zmq
from PIL import Image

from uncertain_worms.utils import PROJECT_ROOT


def main():
    # Set up ZMQ context and a REQ (request) socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    # Load a test image using PIL (replace with your image path)
    image_path = os.path.join(PROJECT_ROOT, "environments/spot", "test_image.jpg")
    try:
        image = Image.open(image_path)
    except IOError:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert image to RGB (in case it's not already)
    image_rgb = image.convert("RGB")

    # Create a dummy depth image with the same height and width as the RGB image.
    depth_image = np.ones((image_rgb.height, image_rgb.width), dtype=np.float32)

    # Convert the PIL image to a NumPy array
    image_array = np.array(image_rgb)
    print("RGB image shape:", image_array.shape)

    # Prepare the message; converting the arrays to lists ensures JSON-serializability.
    message = {
        "rgbPixels": image_array.tolist(),
        "depthPixels": depth_image.tolist(),
        "categories": ["tape measure"],
    }

    print("Sending request to server...")
    message = {"rgbPixels": image_rgb.tolist(), "categories": ["tape measure"]}
    socket.send_json(message)

    # Wait for and process the reply from the server.
    reply = socket.recv_json()
    print("Received reply from server:", reply.keys())

    # Check for the "image" key in the reply
    if "image" in reply:
        # Convert the list to a NumPy array.
        output_image_array = np.array(reply["image"], dtype=np.uint8)
        # Convert the NumPy array to a PIL Image.
        output_image = Image.fromarray(output_image_array)
        # Save the image as a PNG file.
        output_path = os.path.join(PROJECT_ROOT, "server_reply.png")
        output_image.save(output_path, "PNG")
        print(f"Image saved as {output_path}")
    else:
        print("The reply does not contain an 'image' key.")


if __name__ == "__main__":
    main()
