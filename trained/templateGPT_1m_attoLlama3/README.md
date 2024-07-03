# templateGPT_1m_attoLlama3

this model was designed to resemble Llama 3 and be compared against templateGPT_1m_attoGPT2. Notable differences include SwiGLU instead of GeLU, rotary positional encodings, no biases on the linear layers (those two combined means this model actually has ~70k fewer parameters), warm restarts on the learning rate scheduler, and the fact that it's better