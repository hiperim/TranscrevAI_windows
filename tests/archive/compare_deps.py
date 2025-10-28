import re

def parse_requirements(file_path):
    packages = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Handle git dependencies
                if ' @ git+' in line:
                    name = line.split(' @')[0]
                    version = line.split(' @ ')[1]
                    packages[name.lower()] = version
                    continue

                # Use regex to handle ==, >=, etc.
                match = re.match(r"([a-zA-Z0-9\-_\.]+)(.*)", line)
                if match:
                    name = match.group(1).lower()
                    version = match.group(2).strip()
                    packages[name] = version

    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None
    return packages


def main():
    old_packages = parse_requirements('old_requirements.txt')
    new_packages = parse_requirements('requirements.txt')

    if old_packages is None or new_packages is None:
        return

    old_set = set(old_packages.keys())
    new_set = set(new_packages.keys())

    removed_packages = old_set - new_set
    added_packages = new_set - old_set

    modified_packages = {}
    for name in old_set.intersection(new_set):
        if old_packages[name] != new_packages[name]:
            modified_packages[name] = {'old': old_packages[name], 'new': new_packages[name]}

    print("--- Removed Packages ---")
    if removed_packages:
        for pkg in sorted(list(removed_packages)):
            print(pkg)
    else:
        print("None")

    print("\n--- Added Packages ---")
    if added_packages:
        for pkg in sorted(list(added_packages)):
            print(pkg)
    else:
        print("None")

    print("\n--- Modified Packages ---")
    if modified_packages:
        for name, versions in sorted(modified_packages.items()):
            print(f"{name}: {versions['old']} -> {versions['new']}")
    else:
        print("None")

if __name__ == "__main__":
    main()
