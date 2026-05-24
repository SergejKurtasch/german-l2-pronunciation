import json
import sys

source = '/Volumes/SSanDisk/SpeechRec-German/notebooks/03.2.3_g-k_improved_hybrid_cnn_mlp_v4_3_enhanced.ipynb'
target = '/Volumes/SSanDisk/SpeechRec-German-diagnostic/notebooks/03.2.3_schwa-r_improved_hybrid_cnn_mlp_v4_3_enhanced.ipynb'

with open(source, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Replacements
def replace_in_source(source_list):
    if isinstance(source_list, list):
        source_str = ''.join(source_list)
    else:
        source_str = source_list
    source_str = source_str.replace('g/k Phoneme Pair', 'ɐ/ɾ Phoneme Pair')
    source_str = source_str.replace('g/k', 'ɐ/ɾ')
    source_str = source_str.replace("g-k_dl_models_with_context_v2", "schwa-r_dl_models_with_context_v2")
    source_str = source_str.replace("df['class'].isin(['ɡ', 'k'])", "df['class'].isin(['ɐ', 'ɾ'])")
    source_str = source_str.replace("df[df['class'].isin(['ɡ', 'k'])]", "df[df['class'].isin(['ɐ', 'ɾ'])]")
    source_str = source_str.replace("filtering to ɡ/k", "filtering to ɐ/ɾ")
    source_str = source_str.replace("to only ɡ and k classes", "to only ɐ and ɾ classes")
    source_str = source_str.replace('# g=0, k=1', '# ɐ=0, ɾ=1')
    source_str = source_str.replace("np.where(test_preds == 0, 'k', 'ɡ')", "np.where(test_preds == 0, 'ɐ', 'ɾ')")
    source_str = source_str.replace("np.where(val_preds == 0, 'k', 'ɡ')", "np.where(val_preds == 0, 'ɐ', 'ɾ')")
    source_str = source_str.replace('# k=0, ɡ=1', '# ɐ=0, ɾ=1')
    source_str = source_str.replace("k=0, ɡ=1", "ɐ=0, ɾ=1")
    return source_str.splitlines(keepends=True) if isinstance(source_list, list) else source_str

for cell in nb['cells']:
    if 'source' in cell:
        cell['source'] = replace_in_source(cell['source'])

with open(target, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Created: {target}")
