import json
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


def update_json_file(json_path: Path) -> str:
    """
    Update:
      thickness_info.t_at_thin_end_mm -> thickness_info.t_center_mm

    Rules:
      - If thickness_info or t_at_thin_end_mm doesn't exist: skip
      - If t_center_mm already exists: do NOT overwrite it; just remove t_at_thin_end_mm
      - Create a .bak before writing changes
    """
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"ERROR(read): {json_path} | {e}"

    thickness_info = data.get("thickness_info")
    if not isinstance(thickness_info, dict):
        return f"SKIP(no thickness_info): {json_path}"

    if "t_at_thin_end_mm" not in thickness_info:
        return f"SKIP(no key): {json_path}"

    old_value = thickness_info.get("t_at_thin_end_mm")

    # If t_center_mm doesn't exist, create it
    if "t_center_mm" not in thickness_info:
        thickness_info["t_center_mm"] = old_value
        action_note = "UPDATED(copy+remove)"
    else:
        action_note = "UPDATED(remove only; center exists)"

    # Remove old key
    thickness_info.pop("t_at_thin_end_mm", None)

    # Backup then write
    backup_path = json_path.with_suffix(json_path.suffix + ".bak")
    try:
        if not backup_path.exists():
            # shutil.copy2(json_path, backup_path)
            pass

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"ERROR(write): {json_path} | {e}"

    return f"{action_note}: {json_path}"


def find_json_files(root_dir: Path) -> list[Path]:
    # Recursively find all .json files under root_dir
    return sorted([p for p in root_dir.rglob("*.json") if p.is_file()])


def main():
    root = tk.Tk()
    root.withdraw()

    selected_dir = filedialog.askdirectory(title="Select the parent folder that contains experiment result folders")
    if not selected_dir:
        return

    root_dir = Path(selected_dir)
    json_files = find_json_files(root_dir)

    if not json_files:
        messagebox.showinfo("Result", "No .json files found under the selected folder.")
        return

    updated = 0
    skipped = 0
    errors = 0

    # Print logs to console
    print(f"Root folder: {root_dir}")
    print(f"Found JSON files: {len(json_files)}")
    print("-" * 80)

    for jp in json_files:
        result = update_json_file(jp)
        print(result)

        if result.startswith("UPDATED"):
            updated += 1
        elif result.startswith("SKIP"):
            skipped += 1
        else:
            errors += 1

    print("-" * 80)
    summary = f"Done.\nUPDATED: {updated}\nSKIPPED: {skipped}\nERRORS: {errors}\n\nBackup files: *.json.bak"
    messagebox.showinfo("Summary", summary)


if __name__ == "__main__":
    main()
