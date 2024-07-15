
# POST

# def bbox_to_rect(bbox, color):
#     """Convert bounding box to matplotlib format."""
#     # Convert the bounding box (upper-left x, upper-left y, lower-right x,
#     # lower-right y) format to the matplotlib format: ((upper-left x,
#     # upper-left y), width, height)
#     return d2l.plt.Rectangle(
#         xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
#         fill=False, edgecolor=color, linewidth=2)