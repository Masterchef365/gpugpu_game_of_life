Swap buffers
Clear writebuffer
Shader (readbuffer -> writebuffer)
Barrier presentimage (PRESENT\_SRC -> GENERAL)
Shader (writebuffer -> presentimage)
Barrier presentimage (GENERAL -> PRESENT\_SRC)
