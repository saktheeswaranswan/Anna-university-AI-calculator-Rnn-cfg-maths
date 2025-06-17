import random

# Generate a random number with digit length from 1 to 24
def generate_variable_digit_number():
    digits = random.randint(1, 24)
    return random.randint(10**(digits - 1), 10**digits - 1)

# Generate 100000 such numbers
def generate_numbers(n=100000):
    return [generate_variable_digit_number() for _ in range(n)]

# Randomly group numbers into 2 to 4 at a time and perform operations
def perform_operations(numbers):
    results = []
    i = 0
    while i < len(numbers) - 1:
        group_size = random.randint(2, 4)  # group of 2 to 4 numbers
        if i + group_size > len(numbers):
            break
        group = numbers[i:i+group_size]
        a = group[0]
        # Start chaining operations
        addition = a
        subtraction = a
        multiplication = a
        division = a
        modulus = a
        skip = False
        for b in group[1:]:
            addition += b
            subtraction -= b
            multiplication *= b
            if b == 0:
                skip = True  # Avoid division/mod by zero
                break
            division //= b
            modulus %= b
        if not skip:
            results.append({
                'Group': ", ".join(str(num) for num in group),
                'Addition': addition,
                'Subtraction': subtraction,
                'Multiplication': multiplication,
                'Division': division,
                'Modulus': modulus
            })
        i += group_size
    return results

# Save results to a file
def write_results_to_file(results, filename="results_100k.txt"):
    with open(filename, "w") as f:
        for i, res in enumerate(results):
            f.write(f"--- Operation Group {i+1} ---\n")
            for key, value in res.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

# MAIN
if __name__ == "__main__":
    print("Generating 100000 numbers...")
    numbers = generate_numbers()
    print("Performing operations on grouped numbers...")
    results = perform_operations(numbers)
    print("Saving results to text file...")
    write_results_to_file(results)
    print("✅ Done! Results saved in results_100k.txt")
