import os
import shutil
import subprocess

# Function to create a folder, copy final.py into it, and run final.py
def run_experiment(p, q):
    folder_name = f"run_{p}_{q}"

    # Create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Copy final.py into the folder
    shutil.copy("final.py", os.path.join(folder_name, "final.py"))
    shutil.copy("POS.mat", os.path.join(folder_name, "POS.mat"))
    shutil.copy(f"output.log", os.path.join(folder_name, f"output_pos_{p}_{q}.log"))

    

# List of p and q values you want to iterate over
p_values = [0.25, 1.0, 4.0]  # Replace with your desired values
q_values = [0.25, 1.0, 4.0]  # Replace with your desired values

# # Iterate over all combinations of p and q
# for p in p_values:
#     for q in q_values:
#         run_experiment(p, q)

# Create the embeddings_generated folder if it doesn't exist
embeddings_folder = "embeddings_generated"
if not os.path.exists(embeddings_folder):
    os.makedirs(embeddings_folder)

# Move and rename POS.txt files to embeddings_generated folder
for p in p_values:
    for q in q_values:
        source_path = f"run_{p}_{q}/POS.txt"
        dest_path = os.path.join(embeddings_folder, f"POS_{p}_{q}.txt")
        shutil.copy(source_path, dest_path)
