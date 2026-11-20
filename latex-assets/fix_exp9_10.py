with open('exp9_10_content.tex', 'r') as f:
    text = f.read()

# Fix danda line
target_danda = r'danda (\lstinline|॥|, \lstinline|।|) boundary preservation'
replacement_danda = r'Devanagari sentence boundary (\textit{danda} / \textit{poorna viram}) preservation'
text = text.replace(target_danda, replacement_danda)

# Fix Devanagari inside python listing
target_hi = 'train_hi = ["मैं कंप्यूटर विज्ञान पढ़ता हूँ।", "आज मौसम बहुत अच्छा है।", "प्राकृतिक भाषा प्रसंस्करण शक्तिशाली है।"]'
replacement_hi = 'train_hi = ["Main computer vigyan padhta hoon.", "Aaj mausam bahut achha hai.", "Prakritik bhasha prasanskaran shaktishali hai."]'
text = text.replace(target_hi, replacement_hi)

with open('exp9_10_content.tex', 'w') as f:
    f.write(text)

print("Fixed exp9_10_content.tex successfully.")
