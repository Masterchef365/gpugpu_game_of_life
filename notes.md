Swap buffers
Clear writebuffer
Shader (readbuffer -> writebuffer)
Barrier presentimage (PRESENT\_SRC -> GENERAL)
Shader (writebuffer -> presentimage)
Barrier presentimage (GENERAL -> PRESENT\_SRC)

Create image views, refer to them with the descriptor sets that get swapped

* Descriptor sets for:
Image input layer
Image output layer
Swapchain image
