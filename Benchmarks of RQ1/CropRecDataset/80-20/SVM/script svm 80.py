import json
import subprocess
import re
import os
import xml.etree.ElementTree as ET

def parse_trace_file(trace_file):
    last_C = None
    with open(trace_file, 'r') as file:
        data = json.load(file)
        if 'fpvars' in data['init']:
            last_C = data['init']['fpvars'][0]
        for transition in data['transitions']:
            if 'fpvars' in transition['state']:
                last_C = transition['state']['fpvars'][0]
    return last_C

def extract_last_vars(trace_file):
    first_vars_value = None
    with open(trace_file, 'r') as file:
        data = json.load(file)
        if 'transitions' in data and data['transitions']:
            first_transition = data['transitions'][0]
            if 'vars' in first_transition['state']:
                first_vars_value = first_transition['state']['vars'][0]
    return first_vars_value

def extract_opt_values(trace_file):
    PmOpt = None
    HpOpt= None
    with open(trace_file, 'r') as file:
        data = json.load(file)
        if 'transitions' in data and len(data['transitions']) > 0:
            last_transition = data['transitions'][-1]
            if 'state' in last_transition and 'fpvars' in last_transition['state']:
                fpvars = last_transition['state']['fpvars']
                if len(fpvars) > 7:
                    PmOpt = fpvars[6]
                    HpOpt = fpvars[7]
    return PmOPt, HpOpt

def modify_and_run_svm(current_c_value, svma_file, xml_file, last_vars_value, trace_file_path):
    if not os.path.exists(svma_file):
        print(f"Error: {svma_file} not found.")
        return None
    
    with open(svma_file, 'r') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        if 'C =' in line:
            lines[i] = f'C = {current_c_value}\n'
            break
    
    with open(svma_file, 'w') as file:
        file.writelines(lines)
    
    output = subprocess.run(['python', svma_file], capture_output=True, text=True)
    
    accuracy_match = re.search(r'Accuracy:\s+(\d+\.\d+)', output.stdout)
    if accuracy_match:
        Pm_value = float(accuracy_match.group(1))
    else:
        print("Error: Unable to find Pm in SVM output.")
        return None
    
    Pm_value, Hp_value = extract_opt_values(trace_file_path)
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for template in root.iter('template'):
        for declaration in template.iter('declaration'):
            if 'int PmOpt' in declaration.text:
                declaration.text = re.sub(r'int PmOpt\s*=\s*\d+\.\d+;', f'int PmOpt = {Pm_value};', declaration.text)
            if 'int HpOpt' in declaration.text:
                declaration.text = re.sub(r'int HpOpt\s*=\s*\d+\.\d+;', f'int HpOpt = {Hp_value};', declaration.text)
            if 'int C' in declaration.text:
                declaration.text = re.sub(r'int C\s*=\s*\d+\.\d+;', f'int C = {current_c_value};', declaration.text)
            if 'int Pm' in declaration.text:
                declaration.text = re.sub(r'int Pm\s*=\s*\d+\.\d+;', f'int Pm = {Pm_value};', declaration.text)
            if 'int it' in declaration.text:
                for j, line in enumerate(declaration.text.split('\n')):
                    if 'int it' in line:
                        declaration.text = re.sub(r'int it\s*=\s*\d+;', f'int it = {last_vars_value};', declaration.text)
                        break
    
    tree.write(xml_file)
    with open(xml_file, 'r') as f:
        updated_xml_content = f.read()
    print("Updated content of modele.xml:\n", updated_xml_content)
    
    return output.stdout

def main():
    trace_file = r'C:\Users\SYRINE\Desktop\PhD\J1\Crop_recommendation\trace.uctr'
    xml_file = r'C:\Users\SYRINE\Desktop\PhD\J1\Crop_recommendation\modele.xml'
    svma_file = r'C:\Users\SYRINE\Desktop\PhD\J1\Crop_recommendation\svm 1.py'
    
    C = parse_trace_file(trace_file)
    
    if C is not None:
        print("Extracted C value from trace.uctr:", C)
        Cref = 0.68
        
        if C == Cref:
            print("C value equals Cref. No action required.")
        else:
            first_vars_value = extract_last_vars(trace_file)
            print("Last 'it' value:", first_vars_value)
            PmOpt_value, HpOpt_value = extract_opt_values(trace_file)
            print("Extracted PmOpt value:", Pm_value)
            print("Extracted hpOpt value:", Hp_value)
    
            print(f"Running SVM with modified C value ({C})...")
            svm_output = modify_and_run_svm(C, svma_file, xml_file, first_vars_value, trace_file) 
            if svm_output:
                print("SVM execution completed.")
                print("SVM execution results:")
                print(svm_output)
            else:
                print("Error during SVM execution.")
    
    else:
        print("Error: Unable to find C value in trace.uctr.")

if __name__ == "__main__":
    main()
