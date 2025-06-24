def encrypt(text, shift):
    result = ""
    for char in text:
        if char.isalpha():
            # Shift within alphabet (A-Z or a-z)
            base = ord('A') if char.isupper() else ord('a')
            result += chr((ord(char) - base + shift) % 26 + base)
        else:
            result += char  # Leave non-alphabet characters unchanged
    return result

def decrypt(text, shift):
    return encrypt(text, -shift)

# 🖥️ User input
message = input("Enter your message: ")
shift = int(input("Enter shift value: "))

# 🔐 Encrypt and 🔓 Decrypt
encrypted = encrypt(message, shift)
decrypted = decrypt(encrypted, shift)

print("Encrypted message:", encrypted)
print("Decrypted message:", decrypted)
