Swap buffers
Clear writebuffer
Shader (readbuffer -> writebuffer)
Barrier presentimage (PRESENT\_SRC -> GENERAL)
Shader (writebuffer -> presentimage)
Barrier presentimage (GENERAL -> PRESENT\_SRC)

Make the pipeline synchronous: 
* Just one command buffer, image, semaphore per step, etc.
* Wait for queue idle before next
