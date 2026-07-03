import numpy as np

# Replace with the path to your .npy file
data = np.load("/Users/user/Downloads/bottle.npy", allow_pickle=True)

print("Type:", type(data))
print("Shape:", data.shape)

# Try printing the first element
print("\nFirst element:")
print(data[0])

# If it's a dictionary/object
try:
    print("\nKeys:")
    print(data.item().keys())
except Exception:
    pass