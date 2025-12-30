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
                if len(vars_values) > 4:
                    k_value = vars_values[1]
                    it_value = vars_values[-2]
                if len(fpvars_values) > 3:
                    Pm0_value = fpvars_values[0]
    return k_value, it_value, Pm0_value

def modify_and_run_knn(k_value, it_value, Pm0_value, xml_file):
    knn_file = r'C:\Users\SYRINE\Desktop\KNN\KnnNew.py'

    if not os.path.exists(knn_file):
        print("Error: KnnNew.py not found at", knn_file)
        return None
    
    with open(knn_file, 'r') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        if 'knn = KNeighborsClassifier' in line:
            lines[i] = f'knn = KNeighborsClassifier(n_neighbors={k_value})\n'
        elif 'int it =' in line:
            lines[i] = f'int it = {it_value};\n'
    
    with open(knn_file, 'w') as file:
        file.writelines(lines)
    
    output = subprocess.run(['python', knn_file], capture_output=True, text=True)
    
    accuracy_match = re.search(r'Accuracy:\s+(\d+\.\d+)', output.stdout)
    if accuracy_match:
        accuracy_value = float(accuracy_match.group(1))
    else:
        print("Error: Unable to find accuracy in KNN output.")
        return None
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for declaration in root.iter('declaration'):
        declaration.text = re.sub(r'int Pm\s*=\s*\d+\.\d+;', f'int Pm = {Pm_value};', declaration.text)
        declaration.text = re.sub(r'int Pm0\s*=\s*\d+\.\d+;', f'int Pm0 = {Pm0_value};', declaration.text)
        declaration.text = re.sub(r'int Hp\s*=\s*\d+;', f'int Hp = {k_value};', declaration.text)
        declaration.text = re.sub(r'int it\s*=\s*\d+;', f'int it = {it_value};', declaration.text)
    
    tree.write(xml_file)
    
    return output.stdout

def main():
    trace_file = r'C:\Users\SYRINE\Desktop\KNN\traceKNN.uctr'
    xml_file = r'C:\Users\SYRINE\Desktop\KNN\modeleKNN.xml'
    
    k_value, it_value, Pm0_value = parse_trace_file(trace_file)
    
    if k_value is not None and it_value is not None and Pm0_value is not None:
        print("Extracted k value from trace.uctr:", k_value)
        print("Extracted it value from trace.uctr:", it_value)
        print("Extracted accuracy from trace.uctr:", Pm0_value)
        
        print(f"Running KNN with modified k value ({k_value})...")
        knn_output = modify_and_run_knn(k_value, it_value, Pm0_value, xml_file)
        
        if knn_output:
            print("KNN execution completed.")
            print("KNN output results:")
            print(knn_output)
        else:
            print("Error during KNN execution.")
    
    else:
        print("Error: Unable to find k or it value in trace.uctr.")

if __name__ == "__main__":
    main()
