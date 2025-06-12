"""Helper script to compare results with ground truth files."""

import os

def compare_results_to_ground_truth(results, ground_truth):
    """ compare if files contain the same results """
    if len(results) != len(ground_truth):
        return False
    
    for result, gt in zip(results, ground_truth):
        if result.strip() != gt.strip():
            return False

    return True

def main() -> int:
    """
    Main function to run the script.
    """
    TASK_ID = 4
    GROUND_TRUTH_FOLDER = f"train/Task{TASK_ID}/ground-truth"
    PREDICTED_FOLDER = f"train/Task{TASK_ID}/results"

    ground_truth_files = [f for f in os.listdir(GROUND_TRUTH_FOLDER) if f.endswith("_gt.txt")]
    predicted_files = [f for f in os.listdir(PREDICTED_FOLDER) if f.endswith("_results.txt")]
    
    predicted_files.sort()
    ground_truth_files.sort()
    
    for predicted_file in predicted_files:
        predicted_file_name = predicted_file.split("_")
        predicted_file_name = "_".join(predicted_file_name[:-1])
        
        for ground_truth_file in ground_truth_files:
            ground_truth_file_name = ground_truth_file.split("_")
            ground_truth_file_name = "_".join(ground_truth_file_name[:-1])

            if predicted_file_name == ground_truth_file_name:
                with open(os.path.join(GROUND_TRUTH_FOLDER, ground_truth_file), 'r') as gt_file:
                    gt_lines = gt_file.readlines()

                with open(os.path.join(PREDICTED_FOLDER, predicted_file), 'r') as pred_file:
                    pred_lines = pred_file.readlines()

                if compare_results_to_ground_truth(pred_lines, gt_lines):
                    print(f"Results for {predicted_file_name} match the ground truth.")
                else:
                    print(f"Results for {predicted_file_name} do not match the ground truth.")
                break

if __name__ == "__main__":
    exit(main())
