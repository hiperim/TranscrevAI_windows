#!/usr/bin/env python3
"""
Remove classes duplicadas do performance_optimizer.py
Mantém apenas a primeira ocorrência de cada classe
"""

def remove_duplicates():
    input_file = r"C:\TranscrevAI_windows\src\performance_optimizer.py.before_cleanup"
    output_file = r"C:\TranscrevAI_windows\src\performance_optimizer.py"

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"[INFO] Total de linhas: {len(lines)}")

    # Classes a procurar duplicatas
    classes_to_track = [
        'QueueManager',
        'ProcessMonitor',
        'CPUCoreManager',
        'ResourceManager',
        'MultiProcessingTranscrevAI'
    ]

    # Rastrear primeira ocorrência
    first_occurrence = {}
    duplicate_ranges = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Verificar se é declaração de classe
        if stripped.startswith('class '):
            class_name = stripped.split('(')[0].split(':')[0].replace('class ', '').strip()

            if class_name in classes_to_track:
                if class_name not in first_occurrence:
                    # Primeira ocorrência - marcar
                    first_occurrence[class_name] = i
                    print(f"[KEEP] {class_name} primeira ocorrência na linha {i+1}")
                else:
                    # Duplicata - encontrar fim da classe
                    print(f"[DUPLICATE] {class_name} duplicado na linha {i+1}")

                    # Encontrar fim da classe (próxima classe ou função no nível raiz)
                    end_i = i + 1
                    while end_i < len(lines):
                        next_line = lines[end_i].strip()
                        # Próxima definição no nível raiz (sem indentação)
                        if next_line and not lines[end_i].startswith((' ', '\t')):
                            if next_line.startswith('class ') or next_line.startswith('def ') or next_line.startswith('@dataclass'):
                                break
                        end_i += 1

                    duplicate_ranges.append((i, end_i))
                    print(f"  - Marcando linhas {i+1} a {end_i} para remoção")
                    i = end_i
                    continue

        i += 1

    # Remover duplicatas
    lines_to_keep = []
    skip_until = -1

    for i, line in enumerate(lines):
        # Verificar se estamos dentro de um range de duplicata
        if i < skip_until:
            continue

        # Verificar se esta linha inicia uma duplicata
        should_skip = False
        for start, end in duplicate_ranges:
            if i == start:
                skip_until = end
                should_skip = True
                break

        if not should_skip:
            lines_to_keep.append(line)

    # Remover código deprecated (EnhancedTranscrevAIWithMultiprocessing)
    final_lines = []
    for line in lines_to_keep:
        if 'class EnhancedTranscrevAIWithMultiprocessing' in line:
            print(f"[REMOVE] Parando na classe deprecated EnhancedTranscrevAIWithMultiprocessing")
            break
        final_lines.append(line)

    # Escrever resultado
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(final_lines)

    print(f"\n[OK] Remocao de duplicatas concluida!")
    print(f"  - Linhas originais: {len(lines)}")
    print(f"  - Linhas finais: {len(final_lines)}")
    print(f"  - Linhas removidas: {len(lines) - len(final_lines)}")

if __name__ == "__main__":
    remove_duplicates()