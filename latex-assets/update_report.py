import os, sys

def append_section(section_content):
    with open('report.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    end_doc = '\\end{document}'
    if end_doc not in content:
        raise ValueError("Could not find \\end{document} in report.tex")
    
    idx = content.rfind(end_doc)
    new_content = content[:idx] + '\n\n' + section_content.strip() + '\n\n' + content[idx:]
    
    with open('report.tex', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully appended content. New report.tex length:", len(new_content))

if __name__ == '__main__':
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            sec = f.read()
        append_section(sec)
    else:
        print("Usage: python3 update_report.py <section_file.tex>")
