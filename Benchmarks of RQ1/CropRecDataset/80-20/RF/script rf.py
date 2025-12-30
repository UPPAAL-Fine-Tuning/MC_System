import json
import subprocess
import re
import os
import xml.etree.ElementTree as ET

def parse_trace_file(trace_file):
    NE_value = None
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
                    NE_value = vars_values[1]
                    it_value = vars_values[-2]
                if len(fpvars_values) > 3:
                    Pm0_value = fpvars_values[0]
    return NE_value, it_value, Pm0_value

def modify_and_run_rf(NE_value, it_value, Pm0_value, rf_file):
    if not os.path.exists(rf_file):
        print("Error: RandomForestCode.py not found at", rf_file)
        return None
    
    with open(rf_file, 'r') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        if 'n_estimators =' in line:
            lines[i] = f'n_estimators = {NE_value}\n'
            break
    
    with open(rf_file, 'w') as file:
        file.writelines(lines)
    
    output = subprocess.run(['python', rf_file], capture_output=True, text=True)
    
    print(output.stdout)
    
    accuracy_match = re.search(r'Accuracy:\s+(\d+\.\d+)', output.stdout)
    if accuracy_match:
        Pm_value = float(accuracy_match.group(1))
    else:
        print("Error: Unable to find accuracy in RandomForestCode.py output.")
        return None
    
    return Pm_value

def main():
    trace_file = r'C:\Users\SYRINE\Desktop\Crop_recommendation\traceRF.uctr'
    rf_file = r'C:\Users\SYRINE\Desktop\Crop_recommendation\RF 1.py'
    xml_file = r'C:\Users\SYRINE\Desktop\Crop_recommendation\modeleRF.xml'
    
    NE_value, it_value, Pm0_value = parse_trace_file(trace_file)
    
    if NE_value is not None and it_value is not None and Pm0_value is not None:
        print("Extracted NE value from traceRF.uctr:", NE_value)
        print("Extracted it value from traceRF.uctr:", it_value)
        print("Extracted Pm0 value from traceRF.uctr:", Pm0_value)
        
        print(f"Running RandomForest.py with modified n_estimators ({NE_value})...")
        Pm_value = modify_and_run_rf(NE_value, it_value, Pm0_value, rf_file)
        
        if Pm_value is not None:
            print("RandomForestCode.py execution completed.")
            print(f"Accuracy: {Pm_value:.12f}")
            
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for declaration in root.iter('declaration'):
                declaration.text = re.sub(r'int Pm\s*=\s*\d+\.\d+;', f'int Pm = {Pm_value};', declaration.text)
                declaration.text = re.sub(r'int Pm0\s*=\s*\d+\.\d+;', f'int Pm0 = {Pm0_value};', declaration.text)
                declaration.text = re.sub(r'int Hp\s*=\s*\d+;', f'int Hp = {NE_value};', declaration.text)
                declaration.text = re.sub(r'int it\s*=\s*\d+;', f'int it = {it_value};', declaration.text)
            
            tree.write(xml_file)
        else:
            print("Error during RandomForestCode.py execution.")
    else:
        print("Error: Unable to find NE, it, or Pm0 value in traceRF.uctr.")

if __name__ == "__main__":
    main()
