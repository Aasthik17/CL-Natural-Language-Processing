with open('generate_exp8.py') as f:
    text = f.read()

import re

# Let's find where <extra_id_0> is
text = text.replace(r"\texttt{<extra_id_0>}", r"\lstinline|<extra_id_0>|")
text = text.replace(r"\texttt{<extra_id_1>}", r"\lstinline|<extra_id_1>|")

with open('generate_exp8.py', 'w') as f:
    f.write(text)

print("Updated generate_exp8.py")
