# templateGPT_1m_picoGPT2

this model was designed to resemble GPT-2 and be compared against templateGPT_1m_picoLlama3. Notable differences include GeLU instead of SwiGLU, learned positional embeddings, biases on all the linear layers (those two combined means this model actually has ~70k more parameters), a cosine learning rate schedule without warm restarts, and the fact that it's worse