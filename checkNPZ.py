import numpy as np
import sys
import os

def inspect_npz(file_path):
    print(f"🔍 Inspecting: {file_path}")
    
    if not os.path.exists(file_path):
        print("❌ File not found.")
        return

    try:
        # Use mmap_mode='r' to only read metadata, don't load entire file into memory
        # This is critical for large datasets (like your 1.88 million samples) to prevent freezing
        with np.load(file_path, mmap_mode='r') as data:
            print("-" * 40)
            print(f"{'Key':<15} | {'Shape':<20} | {'Dtype':<10}")
            print("-" * 40)
            
            total_samples = 0
            
            for key in data.files:
                arr = data[key]
                print(f"{key:<15} | {str(arr.shape):<20} | {str(arr.dtype):<10}")
                
                # Usually use 'states' length as total sample count
                if key == 'states':
                    total_samples = arr.shape[0]
            
            print("-" * 40)
            print(f"✅ Total Samples: {total_samples:,}")
            print("-" * 40)

    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    # Default reads merged_replay_buffer.npz, can also pass through command line argument
    target_file = sys.argv[1] if len(sys.argv) > 1 else "merged_replay_buffer.npz"
    inspect_npz(target_file)