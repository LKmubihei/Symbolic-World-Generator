from typing import Dict, Any
import re

class Evaluator:
    """
    Evaluator class for comparing PDDL domains.
    """
    
    def eval(self, original: str, generated: str) -> Dict[str, float]:
        """
        Compare original and generated PDDL domains.
        Returns a dictionary of metrics.
        """
        # Extract predicates
        original_predicates = self._extract_predicates(original)
        generated_predicates = self._extract_predicates(generated)
        
        # Extract actions
        original_actions = self._extract_actions(original)
        generated_actions = self._extract_actions(generated)
        
        # Calculate metrics
        predicate_f1 = self._calculate_f1(original_predicates, generated_predicates)
        action_f1_params = self._calculate_action_f1(original_actions, generated_actions, 'params')
        action_f1_preconds = self._calculate_action_f1(original_actions, generated_actions, 'preconditions')
        action_f1_effect = self._calculate_action_f1(original_actions, generated_actions, 'effects')
        
        return {
            'predicate_f1': predicate_f1,
            'action_f1_params': action_f1_params,
            'action_f1_preconds': action_f1_preconds,
            'action_f1_effect': action_f1_effect
        }
    
    def _extract_predicates(self, domain: str) -> set:
        """Extract predicates from PDDL domain."""
        predicates = set()
        pattern = r":predicates\s*\((.*?)\)"
        matches = re.findall(pattern, domain, re.DOTALL)
        
        for match in matches:
            pred = match.strip()
            if pred:
                predicates.add(pred)
        
        return predicates
    
    def _extract_actions(self, domain: str) -> Dict[str, Dict[str, Any]]:
        """Extract actions from PDDL domain."""
        actions = {}
        pattern = r":action\s+(\w+)\s*:parameters\s*\((.*?)\)\s*:precondition\s*\((.*?)\)\s*:effect\s*\((.*?)\)"
        matches = re.findall(pattern, domain, re.DOTALL)
        
        for name, params, preconds, effects in matches:
            actions[name] = {
                'params': params.strip(),
                'preconditions': preconds.strip(),
                'effects': effects.strip()
            }
        
        return actions
    
    def _calculate_f1(self, original: set, generated: set) -> float:
        """Calculate F1 score between two sets."""
        if not original or not generated:
            return 0.0
            
        precision = len(original & generated) / len(generated)
        recall = len(original & generated) / len(original)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_action_f1(self, original: Dict, generated: Dict, field: str) -> float:
        """Calculate F1 score for a specific action field."""
        original_set = set()
        generated_set = set()
        
        for action_name in original:
            if action_name in generated:
                original_set.add(original[action_name][field])
                generated_set.add(generated[action_name][field])
        
        return self._calculate_f1(original_set, generated_set) 