from gen_table import generate_table_files
from bench import run_benchmark

def main():
    if run_benchmark():
        generate_table_files()
        
if __name__ == "__main__":
    main()
