"""
Complete Step 2 → Step 3 Workflow
1. Organize images and convert XML annotations to COCO
2. Verify detection model can run
3. Prepare Step 3 BEV transformation
"""

import os
import subprocess
import sys


def run_step2_workflow():
    """Execute complete Step 2 workflow"""
    
    print("\n" + "="*70)
    print(" STEP 2: OBJECT DETECTION - COMPLETE WORKFLOW")
    print("="*70)
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    step2_dir = os.path.join(root_dir, 'step2_object_detection')
    
    # Step 2.1: Organize images
    print("\n📁 STEP 2.1: Organizing images...")
    print("-" * 70)
    result = subprocess.run(
        [sys.executable, os.path.join(step2_dir, 'organize_images.py')],
        cwd=root_dir,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"⚠️  Error: {result.stderr}")
        return False
    
    # Step 2.2: Convert XML annotations to COCO
    print("\n🔄 STEP 2.2: Converting XML annotations to COCO JSON...")
    print("-" * 70)
    result = subprocess.run(
        [sys.executable, os.path.join(step2_dir, 'xml_to_coco_converter.py')],
        cwd=root_dir,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"⚠️  Error: {result.stderr}")
        return False
    
    # Step 2.3: Verify detection setup
    print("\n✅ STEP 2.3: Verifying object detection setup...")
    print("-" * 70)
    result = subprocess.run(
        [sys.executable, os.path.join(step2_dir, 'test_object_detection.py')],
        cwd=root_dir,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"⚠️  Error: {result.stderr}")
        return False
    
    print("\n[OK] STEP 2 WORKFLOW COMPLETE!")
    print("   Ready to train: python step2_object_detection/train_object_detection.py")
    
    return True


def setup_step3_workflow():
    """Setup Step 3 BEV transformation"""
    
    print("\n" + "="*70)
    print(" STEP 3: BEV TRANSFORMATION - SETUP")
    print("="*70)
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    step3_dir = os.path.join(root_dir, 'step3_bev_transformation')
    
    # Setup BEV directories
    print("\n📁 Setting up BEV transformation directories...")
    print("-" * 70)
    result = subprocess.run(
        [sys.executable, os.path.join(step3_dir, 'setup_bev.py')],
        cwd=root_dir,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"⚠️  Error: {result.stderr}")
        return False
    
    print("\n[OK] STEP 3 SETUP COMPLETE!")
    print("   Next: Collect natural scene thali images and mark reference points")
    
    return True


def print_next_steps():
    """Print clear next steps for the user"""
    
    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    
    print("""
1️⃣  TRAIN OBJECT DETECTION (Step 2)
    ────────────────────────────────
    Command:
      python step2_object_detection/train_object_detection.py
    
    Or on HPC:
      qsub step2_object_detection/run_detection_training.sh
    
    What it does:
      - Trains Faster R-CNN on your annotated thali images
      - Saves best model to: step2_object_detection/thali_detection/models/
      - Generates training report with loss curves
    
    Expected time:
      ~5-10 minutes per epoch (depending on GPU)


2️⃣  PREPARE STEP 3 (BEV Transformation)
    ──────────────────────────────────────
    What you need:
      - Natural scene thali images (from different angles)
      - 4 reference points per image (corners of thali plate)
    
    Files created:
      - step3_bev_transformation/data/
      - BEV transformation module: step3_bev_transformation/bev_transformation.py
    
    Next:
      - Collect more natural scene thali photos
      - Annotate reference points (tool coming soon)


3️⃣  PUSH TO GITHUB
    ────────────────
    Command:
      git add step2_object_detection step3_bev_transformation
      git commit -m "Add object detection training and BEV transformation setup"
      git push


4️⃣  TRANSFER TO HPC (if training on cluster)
    ──────────────────────────────────────────
    Command:
      rsync -avz step2_object_detection/ ashoka:~/Vision_Khana_Project/step2_object_detection/
      rsync -avz shared/ ashoka:~/Vision_Khana_Project/shared/
    """)


def main():
    print("\\n" + "🚀 " * 20)
    print("RUNNING COMPLETE WORKFLOW: Step 2 → Step 3")
    print("🚀 " * 20)
    
    # Run Step 2 workflow
    if not run_step2_workflow():
        print("[ERROR] Step 2 workflow failed!")
        sys.exit(1)
    
    # Setup Step 3
    if not setup_step3_workflow():
        print("[ERROR] Step 3 setup failed!")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()
    
    print("\n" + "="*70)
    print("[OK] ALL WORKFLOWS COMPLETE - YOU'RE READY TO TRAIN!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
