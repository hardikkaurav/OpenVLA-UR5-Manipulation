from transformers import AutoProcessor

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True
)

print("SUCCESS")
