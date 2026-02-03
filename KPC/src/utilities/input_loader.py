import os

def parse_file(filepath):
    """
    Legge file KPC in formato AMPL/DAT.
    Gestisce sintassi: param n := ...; param : V : p w := ...
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File non trovato: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    # Rimuove caratteri inutili per il parsing numerico
    for char in [':=', ':', ';', '(', ')', ',']:
        content = content.replace(char, ' ')

    tokens = content.split()

    data = {
        "n": 0,
        "capacity": 0,
        "profits": [],
        "weights": [],
        "conflicts": []
    }

    # Dizionari temporanei
    temp_profits = {}
    temp_weights = {}

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Leggi n
        if token == 'n':
            try:
                data["n"] = int(tokens[i + 1])
                i += 2
                continue
            except:
                i += 1

        # Legge capacity (c o C)
        elif token.lower() == 'c':
            try:
                data["capacity"] = int(tokens[i + 1])
                i += 2
                continue
            except:
                i += 1

        # Legge tabella p w (cerca pattern "p" seguito da "w")
        elif token == 'p' and (i + 1 < len(tokens)) and tokens[i + 1] == 'w':
            i += 2  # Salta p e w
            count = 0
            while count < data["n"] and i < len(tokens):
                try:
                    idx = int(tokens[i])
                    p_val = int(tokens[i + 1])
                    w_val = int(tokens[i + 2])
                    temp_profits[idx] = p_val
                    temp_weights[idx] = w_val
                    i += 3
                    count += 1
                except ValueError:
                    break
            continue

        # Legge conflitti (cerca "set" o "E")
        elif token == 'set' or token == 'E':
            i += 1
            while i < len(tokens):
                try:
                    u = int(tokens[i])
                    v = int(tokens[i + 1])
                    data["conflicts"].append((u, v))
                    i += 2
                except ValueError:
                    i += 1  # Salta parole come 'end' o simboli

        else:
            i += 1

    # Converte dizionari in liste ordinate
    if data["n"] > 0:
        data["profits"] = [temp_profits.get(k, 0) for k in range(data["n"])]
        data["weights"] = [temp_weights.get(k, 0) for k in range(data["n"])]

    return data

if __name__ == "__main__":
    print("Questo file è una libreria. Esegui main.py invece.")