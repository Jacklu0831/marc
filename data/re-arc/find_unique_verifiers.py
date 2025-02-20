from collections import defaultdict
import ast
import hashlib

def get_function_to_hashes(file_path):
    with open(file_path, 'r') as f:
        file_content = f.read()
    tree = ast.parse(file_content)
    function_hashes = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_body = ast.unparse(node.body)
            function_hash = hashlib.sha256(function_body.encode('utf-8')).hexdigest()
            function_hashes[function_hash].append(node.name)
    return function_hashes


function_to_hashes = function_hashes = get_function_to_hashes("verifiers.py")
for k, v in function_to_hashes.items():
    print(k, v)

n_function = sum(len(v) for v in function_to_hashes.values())
n_unique_function = len(function_to_hashes)
print(n_function, "functions but only", n_unique_function, "unique")
# print(sum(len(v) for v in function_to_hashes.values()), 'functions')
