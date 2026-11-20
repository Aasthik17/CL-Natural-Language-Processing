with open('exp7_content.tex', 'r') as f:
    lines = f.readlines()

lines[99] = r"    \item \textbf{Integer Indexing and Padding (LSTM \& Transformer):} Text is tokenized into word lists, mapped to unique integer IDs using a 20,000-word vocabulary dictionary (with reserved indices \lstinline|<pad>=0| and \lstinline|<unk>=1|), and truncated/padded to a uniform maximum sequence length of $T = 256$ tokens." + "\n"

with open('exp7_content.tex', 'w') as f:
    f.writelines(lines)

print("Line 100 fixed:", repr(lines[99]))
