import json
import subprocess
import re
import os
import xml.etree.ElementTree as ET

def parse_trace_file(trace_file):
    k_value = None
    it_value = None
    Pm0_value = None
    with open(trace_file, 'r') as file:
        data = json.load(file)
        if 'transitions' in data and data['transitions']:
            first_transition = data['transitions'][0]
            if 'state' in first_transition and 'vars' in first_transition['state'] and 'fpvars' in first_transition['state']:
                vars_values = first_transition['state']['vars']
                fpvars_values = first_transition['state']['fpvars']
                if len(vars_values) > 1:
                    k_value = vars_values[1]
                if len(vars_values) > 2:
                    it_value = vars_values[2]
                if len(fpvars_values) > 0:
                    Pm0_value = fpvars_values[0]
    return k_value, it_value, Pm0_value


def modify_and_run_knn(k_value, it_value, Pm0_value, xml_file):
    knn_file = r'C:\Users\SYRINE\Desktop\Crop_recommendation\knn 1.py'

    if not os.path.exists(knn_file):
        print("Error: knn 1.py not found at", knn_file)
        return None
    
    with open(knn_file, 'r') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        if 'neighbors_to_test =' in line:
            lines[i] = f'neighbors_to_test = {k_value}\n'
    
    with open(knn_file, 'w') as file:
        file.writelines(lines)
    
    try:
        completed_process = subprocess.run(['python', knn_file], capture_output=True, text=True)
        if completed_process.returncode == 0:
            output = completed_process.stdout
            print("Output of KNN 1.py:\n", output)
            accuracy_match = re.search(r'Accuracy:\s+(\d+\.\d+)', output)
            if accuracy_match:
                Pm_value = float(accuracy_match.group(1))
                print(f"Number of neighbors: {k_value} - Accuracy: {Pm_value:.6f}")
                return Pm_value
    except subprocess.CalledProcessError as e:
        print(f"Error running KNN: {e.output}")
        return None


def main():
    trace_file = r'C:\Users\SYRINE\Desktop\Crop_recommendation\traceKNN.uctr'
    xml_file = r'C:\Users\SYRINE\Desktop\Crop_recommendation\modeleKNN.xml'
    
    k_value, it_value, Pm0_value = parse_trace_file(trace_file)
    
    if k_value is not None and it_value is not None and Pm0_value is not None:
        print("Extracted k value from trace.uctr:", k_value)
        print("Extracted it value from trace.uctr:", it_value)
        print("Extracted Pm0 value from trace.uctr:", Pm0_value)
        
        print(f"Running KNN with modified neighbors_to_test ({k_value})...")
        knn_accuracy = modify_and_run_knn(k_value, it_value, Pm0_value, xml_file)
        
        if knn_accuracy is not None:
            print("KNN execution completed successfully.")
            print("Obtained Accuracy:", knn_accuracy)
    else:
        print("Error: Unable to find k, it, or Pm0 value in trace.uctr.")


if __name__ == "__main__":
    main()
