with open('exp9_10_content.tex', 'r') as f:
    text = f.read()

# Replace tikz block with includegraphics
tikz_start = r'\begin{tikzpicture}'
tikz_end = r'\end{tikzpicture}'

start_idx = text.find(tikz_start)
end_idx = text.find(tikz_end) + len(tikz_end)

if start_idx != -1 and end_idx != -1:
    fig_replacement = r'\includegraphics[width=0.95\textwidth]{report_assets/miniproject_pipeline.png}'
    new_text = text[:start_idx] + fig_replacement + text[end_idx:]
    with open('exp9_10_content.tex', 'w') as f:
        f.write(new_text)
    print("Replaced tikz block with includegraphics successfully.")
else:
    print("Tikz block not found.")
