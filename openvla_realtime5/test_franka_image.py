from PIL import Image

from config import OpenVLAConfig
from openvla_policy import OpenVLAPolicy

config = OpenVLAConfig()

# Optional for Franka test
config = OpenVLAConfig()

policy = OpenVLAPolicy(config)
policy.load()

image = Image.open("franka3.jpg").convert("RGB")

result = policy.predict(
    image=image,
   instruction = "move right"
)

print("Action:")
print(result.action)
print("Inference time:", result.inference_time_s)