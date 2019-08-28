
Key components


VulkanStream abstraction. Stream abstraction supports several operations:

- stream->launch(function, args, ..)

This maps to:

- obtain current command buffer
- dispatch
- barrier -> (compute|transfer)


- stream->synchronize();

This maps to

- vkEndCommandBuffer
- vkQueueSubmit
- vkQueueWaitIdle



- stream->copy(from=VULKAN, to=VULKAN)
- stream->copy(from=CPU, to=VULKAN)
- stream->copy(from=VULKAN, to=CPU)

This abstracts over CommandBuffer, CommandBufferPool, Fence, etc.


Then, API command are fairly simple, and mirror existing CUDA design.

The key approach is:

