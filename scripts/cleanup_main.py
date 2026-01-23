
def cleanup_main():
    file_path = "app/main.py"
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 1-based indices to 0-based
    # Keep 1..2530 -> lines[0..2530]
    # Delete 2531..3004 -> lines[2530..3004]
    # Keep 3005..End -> lines[3004..]
    
    # Check boundaries
    print(f"Line 2531 (should be session.add): {lines[2530]}")
    print(f"Line 3005 (should be @app.post /health): {lines[3004]}")
    
    new_lines = lines[:2530] + lines[3004:]
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    
    print(f"Fixed {file_path}. Removed lines 2531-3004.")

if __name__ == "__main__":
    cleanup_main()
