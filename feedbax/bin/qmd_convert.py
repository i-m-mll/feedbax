"""Convert .qmd files to .ipynb or .py files in a given directory.

Written with the help of Claude 3.5 Sonnet.
"""

import argparse
import subprocess
import os
import glob

def convert_qmd(directory, to_py=False, keep_ipynb=False):
    # Find all .qmd files in the directory
    qmd_files = glob.glob(os.path.join(directory, "*.qmd"))
    
    if not qmd_files:
        print(f"No .qmd files found in {directory}")
        return
    
    for qmd_file in qmd_files:
        base_name = os.path.splitext(qmd_file)[0]
        ipynb_file = f"{base_name}.ipynb"
        
        try:
            # Step 1: Convert .qmd to .ipynb using quarto
            print(f"Converting {qmd_file} to {ipynb_file}...")
            subprocess.run(["quarto", "convert", qmd_file], check=True)
            
            if to_py:
                # Step 2: Convert .ipynb to .py using nbconvert
                print(f"Converting {ipynb_file} to .py...")
                subprocess.run(["jupyter", "nbconvert", "--to", "python", ipynb_file], check=True)
                
                # Step 3: Remove the intermediate .ipynb file unless keep_ipynb is True
                if not keep_ipynb:
                    os.remove(ipynb_file)
                    print(f"Successfully converted {qmd_file} to {base_name}.py")
                else:
                    print(f"Successfully converted {qmd_file} to {base_name}.py and kept {ipynb_file}")
            else:
                print(f"Successfully converted {qmd_file} to {ipynb_file}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error converting {qmd_file}: {e}")
        except Exception as e:
            print(f"Unexpected error with {qmd_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert .qmd files to .ipynb or .py files")
    parser.add_argument("directory", help="Directory containing .qmd files")
    parser.add_argument("--to-py", action="store_true", help="Convert to Python files instead of keeping as notebooks")
    parser.add_argument("--keep-ipynb", action="store_true", help="Keep the intermediate .ipynb file when converting to Python")
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return
    
    if args.keep_ipynb and not args.to_py:
        print("Warning: --keep-ipynb has no effect when not using --to-py")
    
    convert_qmd(args.directory, args.to_py, args.keep_ipynb)

if __name__ == "__main__":
    main()