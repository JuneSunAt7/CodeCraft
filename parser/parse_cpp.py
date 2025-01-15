import clang.cindex
import os
import json

clang.cindex.Config.set_library_file(r'C:\Program Files\LLVM\bin\libclang.dll')

def parse_cpp_file(file_path):

    index = clang.cindex.Index.create()
    tu = index.parse(file_path)

    print(f"Parsing file: {file_path}")
    print(f"File: {tu.spelling}")

    extracted_data = []

    def visit_node(node, depth=0):
        if node.kind in {clang.cindex.CursorKind.FUNCTION_DECL, clang.cindex.CursorKind.CLASS_DECL}:
            extracted_data.append({
                'name': node.spelling,
                'kind': str(node.kind),
                'line': node.location.line,
                'column': node.location.column,
                'file': str(node.location.file)
            })
        for child in node.get_children():
            visit_node(child, depth + 1)

    visit_node(tu.cursor)
    return extracted_data
def parse_cpp_directory(directory):
    all_data = []

    for filename in os.listdir(directory):
        if filename.endswith('.cpp') or filename.endswith('.h'):
            file_path = os.path.join(directory, filename)
            extracted_data = parse_cpp_file(file_path)
            all_data.extend(extracted_data)

    return all_data

before_data = parse_cpp_directory('../assets/before/')
after_data = parse_cpp_directory('../assets/after/')

with open('../assets/parsed_before.json', 'w') as json_file:
    json.dump(before_data, json_file, indent=4)

with open('../assets/parsed_after.json', 'w') as json_file:
    json.dump(after_data, json_file, indent=4)