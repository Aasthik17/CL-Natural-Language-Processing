with open('exp9_10_content.tex') as f:
    lines = f.readlines()

in_listing = False
for idx, line in enumerate(lines):
    if r'\begin{lstlisting}' in line:
        in_listing = True
        continue
    if r'\end{lstlisting}' in line:
        in_listing = False
        continue
    if not in_listing:
        import re
        s = re.sub(r'\$[^\$]*\$', '', line)
        s = re.sub(r'\\lstinline\|[^\|]*\|', '', s)
        s = s.replace(r'\_', '')
        if '_' in s:
            print(f"Warning line {idx+1}: unescaped underscore in: {line.strip()}")
        if '<' in s or '>' in s:
            print(f"Warning line {idx+1}: unescaped angle bracket in: {line.strip()}")
print("Scan completed.")
