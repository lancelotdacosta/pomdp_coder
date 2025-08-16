# type:ignore
import os
import time
from collections import defaultdict

import cv2
import imageio
import numpy as np
import torch
import zmq

import uncertain_worms.environments.spot.grounding_sam as gsam
from uncertain_worms.utils import PROJECT_ROOT

grounded_checkpoint = "./checkpoints/groundingdino_swint_ogc.pth"
sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
config = "./checkpoints/config/GroundingDINO_SwinT_OGC.py"
device = "cuda"
model = gsam.load_model(config, grounded_checkpoint, device=device)
box_threshold = 0.40
text_threshold = 0.40
sam_version = "vit_h"

predictor = gsam.SamPredictor(
    gsam.sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device)
)


def detection(camera_image, categpry_names):
    priors = defaultdict(lambda: {})

    # Save the input image.
    image_array = camera_image["rgbPixels"]
    save_folder = os.path.join(PROJECT_ROOT, "environments/spot/segmentation_outputs")
    os.makedirs(save_folder, exist_ok=True)
    image_path = os.path.join(save_folder, "raw_image_{}.png".format(time.time()))
    imageio.imsave(image_path, image_array)

    # Load image for Grounding and SAM.
    image_pil, image_for_gs = gsam.load_image(image_path)
    boxes_filt, pred_phrases = gsam.get_grounding_output(
        model,
        image_for_gs,
        " . ".join(categpry_names),
        box_threshold,
        text_threshold,
        with_logits=False,
        device=device,
    )

    # Read image with OpenCV for the SAM predictor.
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # Scale boxes to the original image size.
    H, W = image_pil.size[1], image_pil.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_filt, image.shape[:2]
    ).to(device)

    if boxes_filt.size(0) == 0:
        print("Warning: No detections were made. Returning empty results.")
        return {"priors": {}}

    # Get masks from the SAM predictor.
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    # If no masks (or boxes) are detected, log a warning, save the image with boxes (if any) and return.
    if masks.shape[0] == 0:
        return {"priors": {}}

    # Draw output image with masks and boxes.
    masks = masks.detach().cpu().numpy()

    # Assume 'img' is your original image (e.g., a NumPy array of shape [H, W, 3])
    for mask, box, label in zip(masks, boxes_filt, pred_phrases):
        # Generate a random RGB color and set the alpha (transparency)
        random_color = np.random.random(3)  # values in [0, 1]
        alpha = 0.6  # blending factor

        # Get the mask dimensions (assuming mask is a tensor or numpy array)
        h, w = mask.shape[-2:]
        # Ensure mask is in shape (h, w)
        mask = mask.reshape(h, w)

        # Create a colored version of the mask (only three channels for RGB)
        colored_mask = (mask[..., None] * random_color * 255).astype(np.uint8)

        # Create a boolean mask where the mask is active (assuming mask values are in [0, 1])
        mask_bool = mask > 0

        # Blend the colored mask with the original image using alpha blending.
        # Only blend where mask is active.
        image[mask_bool] = cv2.addWeighted(
            image[mask_bool], 1 - alpha, colored_mask[mask_bool], alpha, 0
        )

        # Convert the box coordinates to integers (avoid using np.uint8 to prevent clipping)
        x0, y0, x1, y1 = box.detach().cpu().numpy().astype(int)

        print(x0, y0, x1, y1)
        # Draw the rectangle on the image (scale color to 0-255 for cv2)
        box_color = (random_color * 255).astype(int).tolist()
        cv2.rectangle(image, (x0, y0), (x1, y1), box_color, 2)

        # Put the label text above the rectangle
        cv2.putText(
            image, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0, 0, 0], 2
        )

    # Instead of enforcing that each label appears, just warn if a label has no associated box or mask.
    unique_labels = set(pred_phrases)
    label_to_box_mask = {}
    for label in unique_labels:
        label_boxes = [
            box for box, phrase in zip(boxes_filt, pred_phrases) if phrase == label
        ]
        label_masks = [
            mask for mask, phrase in zip(masks, pred_phrases) if phrase == label
        ]
        if label_boxes and label_masks:
            label_to_box_mask[label] = (label_boxes[0], label_masks[0])
        else:
            print(
                f"Warning: No box or mask found for label: {label}. Skipping this label."
            )

    # Process each label: here we simply store the mask as a list.
    for label, (box, mask) in label_to_box_mask.items():
        mask_np = mask.squeeze()  # Remove any extra dimensions.
        priors[label] = mask_np.tolist()

    return {"image": image.tolist(), "priors": priors}


def main():
    # Initialize ZMQ server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    print("Planning server started, waiting for requests...")

    while True:
        print("Waiting for request...")
        # Wait for request
        message = socket.recv_json()

        # Extract camera image and categories from message
        rgb = np.array(message["rgbPixels"]).astype(np.uint8)
        print(rgb.shape)
        camera_image = {
            "rgbPixels": rgb,
        }
        categories = message["categories"]

        # Run detection
        results = dict(detection(camera_image, categories))

        # Send response
        socket.send_json(results)


if __name__ == "__main__":
    main()
