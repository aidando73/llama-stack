
def get_counter():
    with open("counter.txt", "r") as f:
        counter = int(f.read().strip())
    counter += 1
    with open("counter.txt", "w") as f:
        f.write(str(counter))
    return counter - 1

def last_counter():
    with open("counter.txt", "r") as f:
        counter = int(f.read().strip())
    return counter - 1
