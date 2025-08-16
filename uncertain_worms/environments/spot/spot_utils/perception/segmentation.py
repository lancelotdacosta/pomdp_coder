"""Some functions for segmenting images and pointclouds using neural
networks."""

from uncertain_worms.environments.spot.spot_utils.structures.image import RGBDImage

# def scene_segment_image(rgbd: RGBDImage) -> RGBDImage:
#     """Segment an image using the MIT Semseg scene segmentation model."""

#     return RGBDImage(
#         get_semantic_labels(rgbd.rgb),
#         rgbd.depth,
#         rgbd.frame,
#         rgbd.intrinsics,
#         rgbd.depth_scale,
#     )
