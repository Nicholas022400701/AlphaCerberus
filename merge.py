# merge.py
import numpy as np
import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def merge_npz_buffers(input_files, output_file):
    all_states = []
    all_policies = []
    all_values = []

    total_samples = 0
    logging.info(f"Merging {len(input_files)} files into {output_file}")

    for file_path in input_files:
        if not os.path.exists(file_path):
            logging.warning(f"Skipping missing file: {file_path}")
            continue
        
        try:
            logging.info(f"Loading: {file_path}")
            with np.load(file_path) as data:
                states = data['states']
                policies = data['policies']
                values = data['values']
                
                if len(states) == 0: continue
                    
                all_states.append(states)
                all_policies.append(policies)
                all_values.append(values)
                
                total_samples += len(states)
                
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            
    if not all_states:
        logging.error("No data to merge.")
        return

    logging.info("Concatenating arrays...")
    final_states = np.concatenate(all_states, axis=0)
    final_policies = np.concatenate(all_policies, axis=0)
    final_values = np.concatenate(all_values, axis=0)
    
    del all_states, all_policies, all_values

    logging.info(f"Total Samples: {len(final_states)}")
    logging.info(f"Saving to {output_file}...")
    
    np.savez_compressed(
        output_file,
        states=final_states,
        policies=final_policies,
        values=final_values
    )
    logging.info("Merge complete.")

def main():
    parser = argparse.ArgumentParser(description="Merge multiple .npz replay buffers.")
    parser.add_argument("input_files", nargs='+', help="List of .npz files to merge")
    parser.add_argument("-o", "--output", dest="output_file", required=True, help="Output .npz file")

    args = parser.parse_args()

    if args.output_file in args.input_files:
        logging.error("Output file cannot be one of the input files.")
        sys.exit(1)

    merge_npz_buffers(args.input_files, args.output_file)

if __name__ == "__main__":
    main()