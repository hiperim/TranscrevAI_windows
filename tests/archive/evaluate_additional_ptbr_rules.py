# tests/evaluate_additional_ptbr_rules.py
"""
Evaluate potential additional PT-BR correction rules.

This script analyzes common PT-BR patterns and suggests safe rules
that could be added beyond the current 12 rules.

Categories:
1. Additional common elisões (Level 1 safe)
2. Common accent errors not covered
3. Contractions and informal speech patterns
"""

# Current 12 rules (BASELINE)
CURRENT_RULES = {
    "nao": "não",
    "voce": "você",
    "esta": "está",
    "ja": "já",
    "la": "lá",
    "tambem": "também",
    "so": "só",
    "entao": "então",
    "porem": "porém",
    "alem": "além",
    "ate": "até",
    "sao": "são"
}

# Potential additional rules - LEVEL 1 (Zero Ambiguity)
ADDITIONAL_LEVEL1 = {
    # Elisões comuns PT-BR coloquial
    "pra": "para",           # pra casa → para casa
    "pro": "para o",         # pro mercado → para o mercado
    "pras": "para as",       # pras meninas → para as meninas
    "pros": "para os",       # pros meninos → para os meninos
    "pruma": "para uma",     # pruma festa → para uma festa
    "prum": "para um",       # prum evento → para um evento

    "ta": "está",            # ta bom → está bom
    "tava": "estava",        # tava lá → estava lá
    "tava": "estava",        # tava indo → estava indo
    "tao": "tão",            # tão bonito → tão bonito

    "ce": "você",            # ce vai? → você vai?
    "ceis": "vocês",         # ceis vão? → vocês vão?

    "ne": "né",              # ne? → né? (já é informal correto)
    "num": "não",            # num sei → não sei

    # Mais acentos sem ambiguidade
    "aqui": "aqui",          # (sem mudança - verificar se Whisper erra)
    "ali": "ali",            # (sem mudança - verificar se Whisper erra)
    "tambem": "também",      # JÁ COBERTO
    "porem": "porém",        # JÁ COBERTO
    "alem": "além",          # JÁ COBERTO

    # Contrações que Whisper pode expandir incorretamente
    "dela": "dela",          # (verificar se precisa)
    "dele": "dele",          # (verificar se precisa)
    "nela": "nela",
    "nele": "nele",
    "numa": "numa",
    "numas": "numas",
    "nuns": "nuns",
    "dum": "de um",          # dum jeito → de um jeito (elisão)
    "duma": "de uma",        # duma forma → de uma forma
}

# Potential additional rules - LEVEL 2 (Low Ambiguity - Needs Review)
ADDITIONAL_LEVEL2 = {
    # Contextuais mas relativamente seguros
    "tá": "está",            # Whisper já transcreve com acento? Verificar
    "pô": "pô",              # Interjeição (pode já estar correto)
    "né": "né",              # Whisper pode transcrever "ne"
}

# NEVER ADD - Level 3 (High Ambiguity)
NEVER_ADD = {
    "e": "é",                # e (conjunção) vs é (verbo) - MUITO AMBÍGUO
    "da": "dá",              # da (preposição) vs dá (verbo) - MUITO AMBÍGUO
    "de": "dê",              # de (preposição) vs dê (verbo) - MUITO AMBÍGUO
    "a": "à/há",             # a (artigo) vs à (crase) vs há (verbo) - IMPOSSÍVEL
    "tem": "têm",            # tem (singular) vs têm (plural) - CONTEXT-DEPENDENT
    "vem": "vêm",            # vem (singular) vs vêm (plural) - CONTEXT-DEPENDENT

    # Confusões fonéticas (NÃO são erros ortográficos)
    "fascista": "machista",  # Palavras diferentes! Context-dependent!
    "confuso": "Confúcio",   # Adjective vs proper noun - context-dependent
    "você": "quem",          # Palavras diferentes! Context-dependent!
}

def analyze_rules():
    """Analyze and categorize potential new rules."""

    print("\n" + "="*70)
    print("PT-BR CORRECTION RULES - EXPANSION ANALYSIS")
    print("="*70)

    print("\n" + "="*70)
    print("CURRENT 12 RULES (BASELINE)")
    print("="*70)
    print(f"Total: {len(CURRENT_RULES)} rules\n")
    for wrong, correct in sorted(CURRENT_RULES.items()):
        print(f"  {wrong:<15} → {correct:<15} (Level 1 - Safe)")

    print("\n" + "="*70)
    print("ADDITIONAL LEVEL 1 RULES (Zero Ambiguity - Safe to Add)")
    print("="*70)
    print("\nElisões e Contrações Comuns PT-BR:\n")

    # Filter out duplicates with current rules
    new_level1 = {k: v for k, v in ADDITIONAL_LEVEL1.items()
                  if k not in CURRENT_RULES}

    print(f"Total candidates: {len(new_level1)} rules\n")

    # Group by category
    elisoes = {}
    accents = {}
    contractions = {}

    for wrong, correct in new_level1.items():
        # Classify
        if wrong in ["pra", "pro", "pras", "pros", "pruma", "prum",
                     "ta", "tava", "tao", "ce", "ceis", "ne", "num"]:
            elisoes[wrong] = correct
        elif wrong != correct and all(c.isalpha() for c in wrong):
            # Check if it's just accent difference
            from unicodedata import normalize
            normalized_wrong = normalize('NFD', wrong).encode('ascii', 'ignore').decode()
            normalized_correct = normalize('NFD', correct).encode('ascii', 'ignore').decode()
            if normalized_wrong == normalized_correct:
                accents[wrong] = correct
            else:
                contractions[wrong] = correct
        else:
            contractions[wrong] = correct

    print("📌 Elisões e Contrações (ALTO POTENCIAL):")
    for wrong, correct in sorted(elisoes.items()):
        print(f"  {wrong:<15} → {correct:<15}")

    if accents:
        print("\n📌 Acentos Adicionais:")
        for wrong, correct in sorted(accents.items()):
            print(f"  {wrong:<15} → {correct:<15}")

    if contractions:
        print("\n📌 Outras Contrações:")
        for wrong, correct in sorted(contractions.items()):
            print(f"  {wrong:<15} → {correct:<15}")

    print("\n" + "="*70)
    print("⚠️  LEVEL 2 RULES (Needs Manual Review)")
    print("="*70)
    print("\nThese need testing to verify they don't cause false positives:\n")
    for wrong, correct in sorted(ADDITIONAL_LEVEL2.items()):
        print(f"  {wrong:<15} → {correct:<15} (Review needed)")

    print("\n" + "="*70)
    print("❌ NEVER ADD (Level 3 - High Ambiguity)")
    print("="*70)
    print("\nThese WILL cause false positives and break correct transcriptions:\n")
    for wrong, correct in sorted(NEVER_ADD.items()):
        print(f"  {wrong:<15} → {correct:<15} (DANGEROUS)")

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("\n1. TEST the Level 1 elisões first (pra, pro, ta, tava, ce, etc.)")
    print("2. These are zero-ambiguity patterns in PT-BR coloquial")
    print("3. Run test_baseline_train_set.py with expanded rules")
    print("4. Compare accuracy vs current 12 rules baseline")
    print("5. If NO regression: Add to production")
    print("6. If regression: Remove problematic rules")
    print("\n" + "="*70)

    # Suggest implementation
    print("\nSUGGESTED IMPLEMENTATION:")
    print("\nAdd to src/transcription.py _init_ptbr_corrections():")
    print("\n```python")
    print("self.ptbr_corrections = {")
    print("    # Current 12 rules")
    print("    " + ",\n    ".join([f'"{k}": "{v}"' for k, v in sorted(CURRENT_RULES.items())]) + ",")
    print("    # Additional elisões (Level 1)")
    print("    " + ",\n    ".join([f'"{k}": "{v}"' for k, v in sorted(elisoes.items())]))
    print("}")
    print("```")
    print("\n" + "="*70)

    return {
        'current': CURRENT_RULES,
        'level1_new': elisoes,
        'level2': ADDITIONAL_LEVEL2,
        'never_add': NEVER_ADD,
        'total_safe': len(CURRENT_RULES) + len(elisoes)
    }

if __name__ == "__main__":
    result = analyze_rules()
    print(f"\n✅ Total safe rules recommended: {result['total_safe']}")
    print(f"   Current: {len(result['current'])}")
    print(f"   New Level 1: {len(result['level1_new'])}")
    print("\n⚠️  Remember: TEST first, then add to production!")
