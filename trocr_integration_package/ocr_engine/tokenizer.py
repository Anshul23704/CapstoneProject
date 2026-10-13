class CharacterTokenizer:
    """
    Dedicated 39-token character-level tokenizer for Indian License Plate Recognition (ALPR).
    Vocabulary consists of:
      - 0: <pad> (Padding token, masked in CTC/Cross-Entropy losses)
      - 1: <bos> (Beginning of sequence token, starts autoregressive decoding)
      - 2: <eos> (End of sequence token, stops autoregressive decoding)
      - 3 - 12: Digits '0' through '9'
      - 13 - 38: Uppercase letters 'A' through 'Z'
    """
    def __init__(self):
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.char_to_idx = {char: idx + 3 for idx, char in enumerate(chars)}
        self.idx_to_char = {idx + 3: char for idx, char in enumerate(chars)}
        
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        
        self.vocab_size = len(chars) + 3

    def encode(self, text, max_length=12):
        tokens = [self.bos_token_id]
        for char in str(text).upper().replace(" ", ""):
            if char in self.char_to_idx:
                tokens.append(self.char_to_idx[char])
        tokens.append(self.eos_token_id)
        
        if len(tokens) < max_length:
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        else:
            tokens = tokens[:max_length - 1] + [self.eos_token_id]
            
        return tokens

    def decode(self, tokens):
        text = ""
        for token in tokens:
            if token == self.eos_token_id:
                break
            if token not in [self.pad_token_id, self.bos_token_id]:
                text += self.idx_to_char.get(token, "")
        return text
