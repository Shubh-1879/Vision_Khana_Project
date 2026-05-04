"""
Khana Dataset - Test Image Setup
Instructions for preparing 20-30 test images for evaluation
"""

import os

def setup_test_images():
    """Setup directory structure for test images"""

    test_dir = os.path.expanduser('~/Vision_Khana_Project/test_images/')
    os.makedirs(test_dir, exist_ok=True)

    print("TEST IMAGE SETUP")
    print("="*50)
    print(f"Created directory: {test_dir}")
    print()
    print("INSTRUCTIONS:")
    print("1. Add 20-30 test images to this directory")
    print("2. Images should be:")
    print("   - High quality photos of individual food items")
    print("   - From the same 80 classes as training data")
    print("   - Not seen during training/validation")
    print("   - Formats: .jpg, .jpeg, .png")
    print()
    print("3. File naming example:")
    print("   - aloo_gobi_test1.jpg")
    print("   - chana_masala_001.jpeg")
    print("   - biryani_test_photo.png")
    print()
    print("4. After adding images, run:")
    print("   python analyze_results.py")
    print()
    print("5. This will generate leaderboard submission format")

    # List current contents
    if os.path.exists(test_dir):
        files = os.listdir(test_dir)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Current test images: {len(image_files)}")
        if image_files:
            print("Files:", image_files[:5], "..." if len(image_files) > 5 else "")

if __name__ == "__main__":
    setup_test_images()