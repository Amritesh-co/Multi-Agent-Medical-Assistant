import os
import glob
from md2pdf.core import md2pdf

# Base directory
base_dir = "/Users/ignite/College/EL 6th_Sem/Multi-Agent-Medical-Assistant"

# Specific patterns and files to check
patterns = [
    "API_KEYS_LOCATION_REFERENCE.md",
    "API_KEYS_SETUP.md",
    "GEMINI_OPTIMIZATION_SUMMARY.md",
    "MODELS_GUIDE.md",
    "README.md",
    "ZERO_AMBIGUITY_EXTENSION_PLAN.md",
    "agents/README.md",
    "Work/*.md",
    "assets/extra_details.md"
]

files_to_convert = []
for pattern in patterns:
    full_pattern = os.path.join(base_dir, pattern)
    files_to_convert.extend(glob.glob(full_pattern))

# Also search for all md files recursion (as per requirement 3)
# But avoiding duplicates from the list above
all_md_files = glob.glob(os.path.join(base_dir, "**/*.md"), recursive=True)
for f in all_md_files:
    if f not in files_to_convert:
        files_to_convert.append(f)

print(f"Found {len(files_to_convert)} markdown files.")

for md_file in files_to_convert:
    pdf_file = os.path.splitext(md_file)[0] + ".pdf"
    
    if os.path.exists(pdf_file):
        print(f"Skipping {md_file}, PDF already exists.")
        continue
    
    try:
        print(f"Converting {md_file} to {pdf_file}...")
        md2pdf(pdf_file, md_content=None, md_file_path=md_file, css_file_path=None, base_url=None)
        print(f"Successfully converted {md_file}")
    except Exception as e:
        print(f"Failed to convert {md_file}: {e}")

# Confirmation
print("\nFinal PDF Check:")
for md_file in files_to_convert:
    pdf_file = os.path.splitext(md_file)[0] + ".pdf"
    exists = "EXISTS" if os.path.exists(pdf_file) else "MISSING"
    print(f"{pdf_file}: {exists}")

