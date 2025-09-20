from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
import sympy
import re

def extract_final_string(text):
    text = text.replace(',', '').strip()
    m = re.search(r'\\?boxed\{([^}]*)\}', text)
    if m:
        content = m.group(1).strip()
        content = re.sub(r'\\,', '', content)
        content = re.sub(r'\\text\{[^}]*\}', '', content)
        content = re.sub(r'^\\$|^\$|\\$|\$', '', content)
        content = content.strip()
        return content
    m = re.search(r'#{3,}\s*([^\n]+)', text)
    if m:
        content = m.group(1).strip()
        content = re.sub(r'\\,', '', content)
        content = re.sub(r'\\text\{[^}]*\}', '', content)
        content = re.sub(r'^\\$|^\$|\\$|\$', '', content)
        content = content.strip()
        return content

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        content = lines[-1]
        content = re.sub(r'\\,', '', content)
        content = re.sub(r'\\text\{[^}]*\}', '', content)
        content = re.sub(r'^\\$|^\$|\\$|\$', '', content)
        content = content.strip()
        return content
    return ''

def multiple_choice_equal(output_text, true_answer):
    matches = re.findall(r"The correct answer is [A-D]", output_text)
    if len(matches) > 0:
        return matches[-1][-1] == true_answer
    else:
        return False

def check_instruction(output_text):
    return "The correct answer is" in output_text

def is_answer_equal(output_text, true_answer, data_path):
    if "gsm8k" in data_path or "math500" in data_path or "scibench" in data_path:
        true_str = extract_final_string(true_answer)
        output_str = extract_final_string(output_text)
        try:
            true_expr = sympy.sympify(true_str)
            output_expr = sympy.sympify(output_str)
            if true_expr == output_expr:
                return True, True
        except Exception as e:
            pass
        true_str_digits = re.sub(r'[^\d\.]', '', true_str)
        output_str_digits = re.sub(r'[^\d\.]', '', output_str)
        return true_str_digits == output_str_digits, True
    else:
        return multiple_choice_equal(output_text, true_answer), check_instruction(output_text)

def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> float:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score