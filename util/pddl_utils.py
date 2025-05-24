import os
import re
import openai

import time
from pprint import pprint
import argparse
from openai import OpenAI
import sys
import subprocess
from pathlib import Path
# import torch
import Levenshtein
import random
from datetime import datetime
sys.path.insert(0, '/home/lk/symbolic_nips/pddl-parser') # replace with your own path
# print(sys.path)
from pddl_parser.planner import Planner
import json






# 0.读取文件的辅助函数

def read_file(file_path):
  with open(file_path,'r') as f:
    data=f.read()
  return data.strip()


def postprocess_completion_action(completion):
    pattern = re.compile(r"```\S*\s*(\(:action.*?)\s*```", re.DOTALL)
    action_code_blocks = pattern.findall(completion)
    return '\n\n'.join(action_code_blocks)+'\n\n)'

def extract_domain_name(pddl_text):
    pattern = r"\(define \(domain\s+(\S+)"
    match = re.search(pattern, pddl_text)
    if match:
        return match.group(1).replace(')', '')
    else:
        return None

def extract_pddl(text):
    if not isinstance(text, str):
        return False, "Input 'text' must be a string"
    if not text.strip():
        return False, "Input 'text' cannot be empty"
        
    pattern = r"```pddl\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches == []:
        pattern = r"```\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
    
    if not matches:
        return False, "No PDDL code block found in the text"
        
    pddl = max(matches, key=len).replace('```pddl', '').replace('```', '').strip()
    if not pddl:
        return False, "Extracted PDDL code is empty"
        
    return True, pddl




def solve_planning(domain_file_path, problem_file_path):
    # 创建规划器实例
    planner = Planner()
    
    # 自动推导输出文件路径
    # print(os.path.splitext(domain_file_path))
    output_file_path = os.path.splitext(domain_file_path)[0] +"_output.txt"
    
        # 调用 Lapkt 求解器（需要根据你的具体实现来定义）
    def call_lapkt_solver(
        problem_file: str,
        domain_file: str,
        output_file="output.txt" 
    ) :

        docker_repo = "lapkt/lapkt-public"
        command = [
            "docker",
            "run",
            "--rm",  # 运行结束后自动删除容器
            "-v",
            f"{os.path.dirname(domain_file)}:/data", # 将当前工作目录挂载到 Docker 容器中的 /data
            f"{docker_repo}",  # 使用的 Docker 镜像
            "timeout",  # 设置超时命令
            "30s",  # 设置规划器的超时时间（例如300秒）
            # "MAX_VARS",  # 设置超时命令
            # "30",  # 设置规划器的超时时间（例如300秒）
            "./bfs_f",  # 假设规划器可执行文件在容器中的 /planner 路径下
            "--domain", f"/data/{os.path.basename(domain_file)}",  # 传递域文件
            "--problem", f"/data/problems/{os.path.basename(problem_file)}",  # 传递问题文件
            "--output",
            f"/data/{os.path.basename(output_file)}",
        ]

        try:
            # 执行 Docker 命令
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:

                
                return False, str(result.stdout.decode())

            with open(output_file, "r") as file:
                lines = file.readlines()
                content = [
                    line.strip()[1:-1].lower() for line in lines if "REACH-GOAL" not in line
                ]
                return True, content
            
            
        except FileNotFoundError as e:
            print(f"文件未找到: {str(e)}")
            return False, str(e)
        except Exception as e:
            print(f"发生了一个意外错误: {str(e)}")
            return False, str(e)


    # 规划问题，调用 LAPKT 求解器
    success,plan = call_lapkt_solver(problem_file_path, domain_file_path, output_file_path)
    if not success or not plan:
        parser_problem_plan = planner.solve(Path(domain_file_path), Path(problem_file_path))
        if  (not parser_problem_plan) or (isinstance(parser_problem_plan, (TimeoutError,TypeError, AttributeError, ValueError))):
            pass
        else:
            print(f"求解器调用成功: {plan}")
            return True, parser_problem_plan

    return success,plan 


from tarski.io import PDDLReader
from tarski.syntax.formulas import *
import traceback

def checker(_domain, gold_domain, raise_on_error=True):
    try:
        reader = PDDLReader(raise_on_error=True)
        reader_gold = PDDLReader(raise_on_error=True)

        reader.parse_domain(_domain)
        actions = reader.problem.actions

        reader_gold.parse_domain(gold_domain)
        gold_actions = reader_gold.problem.actions

        if set(actions.keys()) != set(gold_actions.keys()):
            print('Actions do not match, the actions in the domain are not the same as the actions in the gold domain:' + str(set(gold_actions.keys()) - set(actions.keys())))
            return False, 'Actions do not match, the actions in the domain are not the same as the actions in the gold domain:' + str(set(gold_actions.keys()) - set(actions.keys()))
        return True, 'Success'
        
    except Exception as e:
        exception_type = type(e).__name__
        traceback_info = traceback.format_exc()
        error_message = f"{exception_type}: {str(e)}"
        print(error_message)
        return False, error_message

def count_atomic_preconditions(precond_str):
    # 假设 precond_str 是类似 "(atom1 and atom2 and ...)" 的字符串
    s = precond_str.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    # 按 " and " 分割（注意：这是一种简单的启发式方法）
    atoms = s.split(" and ")
    # 返回非空原子公式的数量
    return len([atom for atom in atoms if atom.strip() != ""])

def count_atomic_effects(effects):
    cnt = 0
    for eff in effects:
        # 将效果转换为字符串，比如 "(T -> ADD(atlocation(?a,?lstart)))"
        s = str(eff).strip()
        # 去除外层括号
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]
        # 如果格式为 "T -> ...", 则右侧部分为原子效果
        parts = s.split("->")
        if len(parts) > 1:
            # 我们只计数1个原子效果
            cnt += 1
        else:
            cnt += 1
    return cnt

def count_action_atomic_formulas(pddl_domain):
    reader = PDDLReader(raise_on_error=True)
    reader.parse_domain(pddl_domain)
    actions = reader.problem.actions
    total = 0
    for action in actions.values():
        pre_str = str(action.precondition)
        pre_count = count_atomic_preconditions(pre_str)
        eff_count = count_atomic_effects(action.effects)
        total += pre_count + eff_count
    return total


def parse_actions(pddl_domain):
    """Parse domain actions and return a map of action names to parameter counts"""
    # Clean up the domain string
    pddl_domain = pddl_domain.strip()
        
    reader = PDDLReader(raise_on_error=True)
    reader.parse_domain_string(pddl_domain)
        
    return reader.problem.actions

def parse_predicates(pddl_domain):
    """Parse domain predicates and return a map of predicate names to arities"""
    pddl_domain = pddl_domain.strip()
    
    reader = PDDLReader(raise_on_error=True)
    reader.parse_domain_string(pddl_domain)
    predicate_map = {}
    for pred in reader.problem.language.predicates:
        if str(pred.symbol) not in ['=', '!=']:
            predicate_map[str(pred.symbol)] = pred.arity
    return predicate_map

def _purge_comments(pddl_str):
    # Purge comments from the given string
    while True:
        match = re.search(r";(.*)\n", pddl_str)
        if match is None:
            break  # Changed from return to break to handle newlines after
        start, end = match.start(), match.end()
        pddl_str = pddl_str[:start]+pddl_str[end-1:]
    
    # First remove empty lines that only contain whitespace
    pddl_str = re.sub(r'\n\s+\n', '\n\n', pddl_str)
    # Then remove consecutive newlines (more than 2) with just 2 newlines
    pddl_str = re.sub(r'\n{2,}', '\n\n', pddl_str)
    
    return pddl_str

def pddl_tokenize(text):
    text = _purge_comments(text)
    
    pddl_patterns = [
        r'\(\s*define',
        r':domain',
        r':requirements',
        r':types',
        r':predicates',
        r':action',
        r':parameters',
        r':precondition',
        r':effect',
        
        r':constants',
        r':functions',
        r':durative-action',
        r':derived',
        
        r':strips',
        r':typing',
        r':negative-preconditions',
        r':disjunctive-preconditions',
        r':equality',
        r':existential-preconditions',
        r':universal-preconditions',
        r':quantified-preconditions',
        r':conditional-effects',
        r':fluents',
        r':adl',
        r':durative-actions',
        r':derived-predicates',
        
        r'not',
        r'and',
        r'or',
        r'exists',
        r'forall',
        r'when',
        r'imply',
        r'preference',
        
        r'increase',
        r'decrease',
        r'assign',
        r'scale-up',
        r'scale-down',
        
        r'[<>=]=?', 
        
        r'-?\d+\.?\d*',   
        r'#t',           
        r'\?duration',  
        
        r'\?[a-zA-Z][a-zA-Z0-9_-]*',  
        r'[a-zA-Z][a-zA-Z0-9_-]*',    
        r'[!$%&*+./<=>?@^_~-][!$%&*+./<=>?@^_~-]*',
        
        r'\(|\)',
        r'-',
    ]
    
    pattern = '|'.join(pddl_patterns)
    tokens = re.findall(pattern, text, re.IGNORECASE)
    
    tokens = [t.strip().lower() for t in tokens if t.strip()]
    return tokens

def test_extract_pddl():
    gpt_response = ""
    pddl_domain = extract_pddl(gpt_response)
    print(pddl_domain)


class Evaluator:

    def __init__(self):
        self.sim_model = None
        self.file_name = str(datetime.now()) + '_' + str(random.randint(1, 114514))

    def eval(self, gt_domain_text, pred_domain_text): # text
        levenshtein_ratio_cleaned = 0
        action_f1_dict = {'params': [], 'preconds': [], 'effect': []}
        predicate_f1_val = 0


        try:
            action_f1_dict = self.action_f1(gt_domain=gt_domain_text, pred_domain=pred_domain_text)
            predicate_f1_val = self.predicate_f1(gt_domain=gt_domain_text, pred_domain=pred_domain_text)
        except:
            pass

        print("Ground Truth Domain Text:", action_f1_dict)
        print("Predicted Domain Text:", predicate_f1_val)

# text distance

        gt_domain_text_cleaned = _purge_comments(gt_domain_text)
        pred_domain_text_cleaned = _purge_comments(pred_domain_text)
        # Remove all whitespace, newlines, tabs and other meaningless characters for PDDL
        gt_cleaned = re.sub(r'\s+', '', gt_domain_text_cleaned)
        pred_cleaned = re.sub(r'\s+', '', pred_domain_text_cleaned)
        levenshtein_ratio_cleaned = self.cal_Levenshtein_ratio(gt_cleaned, pred_cleaned)


        result = dict()
        result['levenshtein_ratio_cleaned'] = levenshtein_ratio_cleaned
        # result['action_f1'] = action_f1_dict
        result['predicate_f1'] = predicate_f1_val
        result.update({'action_f1_'+k: v for k, v in action_f1_dict.items()})
        # Multiply all numeric values by 100 and round to 1 decimal place
        for key in result:
            if isinstance(result[key], (int, float)):
                result[key] = round(result[key] * 100, 1)
            elif isinstance(result[key], dict):
                for subkey in result[key]:
                    if isinstance(result[key][subkey], (int, float)):
                        result[key][subkey] = round(result[key][subkey] * 100, 1)
        return result
    
    def cal_Levenshtein_ratio(self, text1, text2):
        ratio = Levenshtein.ratio(text1, text2)
        return ratio

    def compute_f1_score(self, prediction, reference):
        if len(prediction) == 0 and len(reference) == 0:
            return 1
        pred_tokens = prediction
        ref_tokens = reference
        
        true_positives = len(set(pred_tokens) & set(ref_tokens))
        false_positives = len(set(pred_tokens) - set(ref_tokens))
        false_negatives = len(set(ref_tokens) - set(pred_tokens))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1


    def predicate_f1(self, gt_domain, pred_domain):
        try:
            gt_pred_dict = parse_predicates(gt_domain)
            pred_pred_dict = parse_predicates(pred_domain)
            gt_predicates = [f"{k}_{v}" for k,v in gt_pred_dict.items()]
            pred_predicates = [f"{k}_{v}" for k,v in pred_pred_dict.items()]
            return self.compute_f1_score(pred_predicates, gt_predicates)
        except Exception as e:  # not executable
            print('error', e)
            return 0
        
    def _preprocess(self, l):
        l = map(str, l)
        l = sorted(l)
        for i in range(len(l)):
            if '(' == l[i][0] and ' or ' in l[i]:
                inner = l[i][1:-1].strip()
                parts = [p.strip() for p in inner.split(' or ')]
                parts = sorted(parts)
                l[i] = '(' + ' or '.join(parts) + ')'
        return sorted(l)


    def action_f1(self, gt_domain, pred_domain):    # param F1, preconds F1, effect F1
        def _mean(l):
            if len(l) == 0:
                return 0
            else:
                return sum(l) / len(l)
        metrics = {'params': [], 'preconds': [], 'effect': []}

        gt_actions = parse_actions(gt_domain)
        gt_actions = {k:{'params': self._preprocess([f"{p.symbol}" for p in v.parameters]),
                            'preconds':self._preprocess([x.strip() for x in str(v.precondition)[1:-1].split('and')]),
                            'effect':self._preprocess([str(x) for x in v.effects])} for k, v in gt_actions.items()}
        pred_actions = parse_actions(pred_domain)
        pred_actions = {k:{'params': self._preprocess([f"{p.symbol}" for p in v.parameters]),
                            'preconds':self._preprocess([x.strip() for x in str(v.precondition)[1:-1].split('and')]),
                            'effect':self._preprocess([str(x) for x in v.effects])} for k, v in pred_actions.items()}

        for k, gt_action in gt_actions.items():
            pred_action = pred_actions[k]
            metrics['params'].append(self.compute_f1_score(pred_action['params'], gt_action['params']))
            metrics['preconds'].append(self.compute_f1_score(pred_action['preconds'], gt_action['preconds']))
            metrics['effect'].append(self.compute_f1_score(pred_action['effect'], gt_action['effect']))

        metrics = {k: _mean(v) for k, v in metrics.items()}
        return metrics


