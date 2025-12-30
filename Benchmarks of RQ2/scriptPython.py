import json
import subprocess
import re
import os
import xml.etree.ElementTree as ET
import random

def generate_random_Hp():
    return random.uniform(0.001, 1000.0) 

def parse_trace_file(trace_file):
    last_Hp = None
    with open(trace_file, 'r') as file:
        data = json.load(file)
        if 'fpvars' in data['init']:
            last_Hp = data['init']['fpvars'][0]
        for transition in data['transitions']:
            if 'fpvars' in transition['state']:
                last_Hp = transition['state']['fpvars'][0]
    return last_Hp

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
    HpOpt = None
    with open(trace_file, 'r') as file:
        data = json.load(file)
        if 'transitions' in data and len(data['transitions']) > 0:
            last_transition = data['transitions'][-1]
            if 'state' in last_transition and 'fpvars' in last_transition['state']:
                fpvars = last_transition['state']['fpvars']
                if len(fpvars) > 7:
                    PmOpt = fpvars[6]
                    HpOpt = fpvars[7]
    return PmOpt, HpOpt

def modify_and_run_svm(current_Hp_value, xml_file, last_vars_value, trace_file_path):
    svma_file = r'C:\Users\SYRINE\Desktop\Random\svmNew.py'
    
    if not os.path.exists(svma_file):
        print("Error: svmNew.py not found at", svma_file)
        return None
    
    with open(svma_file, 'r') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        if 'clf = svm.SVC' in line:
            lines[i] = f'clf = svm.SVC(kernel=\'linear\', C={current_Hp_value})\n'
            break
    
    with open(svma_file, 'w') as file:
        file.writelines(lines)
    
    output = subprocess.run(['python', svma_file], capture_output=True, text=True)
    
    accuracy_match = re.search(r'accuracy:\s+(\d+\.\d+)', output.stdout)
    if accuracy_match:
        accuracy_value = float(accuracy_match.group(1))
    else:
        print("Error: Unable to find Pm in SVM output.")
        return None
    
    PmOpt, HpOpt = extract_opt_values(trace_file_path)
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for template in root.iter('template'):
        for declaration in template.iter('declaration'):
            if 'int PmOpt' in declaration.text:
                declaration.text = re.sub(r'int Pm\s*=\s*\d+\.\d+;', f'int PmOpt = {PmOpt};', declaration.text)
            if 'int HpOpt' in declaration.text:
                declaration.text = re.sub(r'int HpOpt\s*=\s*\d+\.\d+;', f'int HpOpt = {HpOpt};', declaration.text)
            if 'int Hp' in declaration.text:
                declaration.text = re.sub(r'int C\s*=\s*\d+\.\d+;', f'int Hp = {current_Hp_value};', declaration.text)
            if 'int Pm' in declaration.text:
                declaration.text = re.sub(r'int Pm\s*=\s*\d+\.\d+;', f'int Pm = {Pm_value};', declaration.text)
            for j, line in enumerate(declaration.text.split('\n')):
                if 'int it' in line:
                    declaration.text = re.sub(r'int it\s*=\s*\d+;', f'int it = {last_vars_value};', declaration.text)
                    break
    
    tree.write(xml_file)
    with open(xml_file, 'r') as f:
        updated_xml_content = f.read()
    print("Updated modele.xml content:\n", updated_xml_content)
    
    return output.stdout

def main():
    trace_file = r'C:\Users\SYRINE\Desktop\Random\trace.uctr'
    xml_file = r'C:\Users\SYRINE\Desktop\Random\modele.xml'
    
    Hp = generate_random_Hp()
    
    if Hp is not None:
        print("Generated random Hp value:", Hp)
        
        first_vars_value = extract_last_vars(trace_file)
        print("Last 'it' value:", first_vars_value)
        
        PmOpt, HpOpt = extract_opt_values(trace_file)
        print("Extracted PmOpt value:", PmOpt)
        print("Extracted HpOpt value:", HpOpt)

        print("Running SVM with random Hp value... (Hp =", Hp, ")")
        svm_output = modify_and_run_svm(Hp, xml_file, first_vars_value, trace_file) 
        if svm_output:
            print("SVM run completed.")
            print("SVM execution output:")
            print(svm_output)
        else:
            print("Error during SVM execution.")
    else:
        print("Error: Unable to generate a random Hp value.")

if __name__ == "__main__":
    main()
