Create image views, refer to them with the descriptor sets that get swapped

* Descriptor sets for:
Image input layer
Image output layer
Swapchain image

Barrier (intermediate `UNDEFINED` -> `GENERAL`)
Dispatch ( -> intermediate)
Barrier (intermediate `GENERAL` -> `TRANSFER_SRC`), (swapchain `UNDEFINED` -> `TRANSFER_DST`)
CopyImage (intermediate -> swapchain)
Barrier (swapchain `TRANSFER_DST` -> `PRESENT_SRC`)

Create image view for intermediate
