import numpy as np

def gen_text(prompt: str, n_tokens_to_generate: int = 40):
    """
    Упрощённая GPT-2-like генерация, согласованная с тестовой заготовкой.
    Компоненты:
      • Token Embeddings (wte) + Positional Embeddings (wpe)
      • Финальный LayerNorm (из params['ln_f']), здесь фактически тождество
      • Автогрегрессия с argmax по логитам (weight tying к wte)

    ВНИМАНИЕ: преднамеренно не добавляем случайные веса блоков (attention/FFN),
    чтобы поведение полностью определялось wte/wpe из load_encoder_hparams_and_params
    и было воспроизводимо через np.random.seed(...).
    """
    # --- Вспомогательные функции ---
    def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Layer Normalization по последней оси.
        x: (d,), gamma: (d,), beta: (d,)
        Возвращает: (d,)
        """
        mean = x.mean(keepdims=True)
        var = x.var(keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)
        return x_hat * gamma + beta

    # --- Загружаем «энкодер», гиперпараметры и параметры модели ---
    encoder, hparams, params = load_encoder_hparams_and_params()
    wte: np.ndarray = params["wte"]          # (vocab_size, d_model)
    wpe: np.ndarray = params["wpe"]          # (n_ctx, d_model)
    ln_g: np.ndarray = params["ln_f"]["g"]   # (d_model,)
    ln_b: np.ndarray = params["ln_f"]["b"]   # (d_model,)

    d_model = wte.shape[1]

    # --- Токенизируем prompt ---
    input_ids = encoder.encode(prompt)  # список int
    if len(input_ids) == 0:
        # Если пустой prompt — начинаем с <UNK>, чтобы было от чего генерировать
        input_ids = [encoder.encoder_dict["<UNK>"]]

    generated_ids = []

    # --- Автогрегрессивная генерация ---
    for _ in range(n_tokens_to_generate):
        T = len(input_ids)
        last_id = input_ids[-1]                # последний токен в текущем контексте

        # Вектор последней позиции: токен + позиция
        # x_last: (d_model,)
        x_last = wte[last_id] + wpe[T - 1]

        # Финальный LayerNorm (в наших params это тождество, но оставим для полноты)
        h_last = layer_norm(x_last, ln_g, ln_b)

        # Логиты по словарю через привязку весов к wte (weight tying)
        # logits: (vocab_size,)
        logits = h_last @ wte.T

        # Выбираем следующий токен детерминированно (argmax)
        next_id = int(np.argmax(logits))

        # Обновляем контекст и список сгенерированных токенов
        input_ids.append(next_id)
        generated_ids.append(next_id)

    # --- Декодируем ТОЛЬКО сгенерированные токены (prompt не включаем) ---
    return encoder.decode(generated_ids)

def load_encoder_hparams_and_params(model_size: str = "124M", models_dir: str = "models"):
	class DummyBPE:
		def __init__(self):
			self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

		def encode(self, text: str):
			tokens = text.strip().split()
			return [self.encoder_dict.get(token, self.encoder_dict["<UNK>"]) for token in tokens]

		def decode(self, token_ids: list):
			reversed_dict = {v: k for k, v in self.encoder_dict.items()}
			return " ".join([reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids])

	hparams = {
		"n_ctx": 1024,
		"n_head": 12
	}

	params = {
		"wte": np.random.rand(3, 10),
		"wpe": np.random.rand(1024, 10),
		"blocks": [],
		"ln_f": {
			"g": np.ones(10),
			"b": np.zeros(10),
		}
	}

	encoder = DummyBPE()
	return encoder, hparams, params

np.random.seed(42) 
print("Generating text with simplified GPT-2-like model...")
print("Test 1")
print(gen_text("hello world", n_tokens_to_generate=10))
print("\nTest 2")
print(gen_text("hello", n_tokens_to_generate=5))
print("\nTest 3")
print(gen_text("world", n_tokens_to_generate=3))