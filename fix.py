import re

with open('src/opera/discovery/pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'\"\"\"(.*?)\"\"\"', '\"\"\"\"\"\"', content, flags=re.DOTALL)

with open('src/opera/discovery/pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)
