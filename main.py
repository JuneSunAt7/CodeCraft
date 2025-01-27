import pandas as pd

# Создаем список кода и соответствующих меток
data = {
    'code': [
        '#include <iostream>\nint main() {\nstd::cout << "Hello, World!" << std::endl;\nreturn 0;\n}',  # хороший код
        '#include <iostream>\nint main() {\nstd::cout << "Missing semicolon"\nreturn 0;\n}',  # синтаксическая ошибка
        '#include <iostream>\nint main() {\nint a = 5;\nint b = 0;\nstd::cout << a / b << std::endl;\nreturn 0;\n}',  # логическая ошибка
        '#include <iostream>\nint main() {\nint a = 10;\nstd::cout << a << std::endl;\n}',  # хороший код
        '#include <iostream>\nint main() {\nint a = 10;\nif (a = 5) std::cout << "Error";\nreturn 0;\n}',  # логическая ошибка
        '#include <iostream>\nint main() {\nstd::cout << "No main function";\n}',  # синтаксическая ошибка
        '#include <iostream>\nint main() {\nstd::cout << "Valid code!" << std::endl;\nreturn 0;\n}',  # хороший код
        '#include <iostream>\nint main() {\n  int a = 10;\n  return;\n}',  # синтаксическая ошибка
        '#include <iostream>\nint main() {\nint a = 10;\nint b = 0;\nstd::cout << a / b << std::endl;\n}',  # логическая ошибка
        '#include <iostream>\nint add(int x, int y) {\nreturn x + y;\n}\nint main() {\nstd::cout << add(5, 10);\nreturn 0;\n}'  # хороший код
    ],
    'label': [
        0,  # хороший код
        1,  # синтаксическая ошибка
        2,  # логическая ошибка
        0,  # хороший код
        2,  # логическая ошибка
        1,  # синтаксическая ошибка
        0,  # хороший код
        1,  # синтаксическая ошибка
        2,  # логическая ошибка
        0   # хороший код
    ]
}

# Создаем DataFrame из данных
df = pd.DataFrame(data)

# Сохраняем DataFrame в CSV файл
df.to_csv('assets/gen.csv', index=False)