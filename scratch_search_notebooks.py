import os
import json

notebooks_dir = "./notebooks"
if os.path.exists(notebooks_dir):
    for f in os.listdir(notebooks_dir):
        if f.endswith(".ipynb"):
            path = os.path.join(notebooks_dir, f)
            with open(path, "r", encoding="utf-8") as file:
                content = json.load(file)
            for i, cell in enumerate(content.get("cells", [])):
                if cell.get("cell_type") == "code":
                    for output in cell.get("outputs", []):
                        text = "".join(output.get("text", []))
                        if "721" in text and "387" in text:
                            print(f"Found in notebook: {f}, cell: {i}")
                            print(text[:1000])
else:
    print("Notebooks directory not found")
