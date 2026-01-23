
def fix_indent():
    file_path = "app/main.py"
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 0-indexed: line 2321 is index 2320.
    start_idx = 2320 
    end_idx = 2506 # Approximation, check end of function.
    
    print(f"Checking line {start_idx+1}: {repr(lines[start_idx])}")
    
    # Ensure we are targeting the right block
    if "with get_session() as session:" not in lines[start_idx]:
        print("Error: Target line mismatch. Aborting.")
        return

    # Dedent loop
    for i in range(start_idx, len(lines)):
        # Stop if we hit the next function definition (def ...)
        # or if we go too far.
        # But wait, next function is `def _state_snapshot`.
        if lines[i].startswith("def _state_snapshot"):
            end_idx = i
            break
        
        # Only dedent if it starts with spaces
        if lines[i].startswith("        "): # 8 spaces
            lines[i] = lines[i][4:] # Remove first 4 spaces
        elif lines[i].strip() == "": # Empty line
            pass # Keep as is (or strip, but easy to leave)
    
    print(f"Dedented lines {start_idx+1} to {end_idx}.")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    print("Fix applied.")

if __name__ == "__main__":
    fix_indent()
